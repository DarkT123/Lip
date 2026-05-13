"""Entry point: hotkey-driven inference loop.

Default mode (transcribe-only):
    Press ⌘⇧L → record → AVSR transcribe → display + clipboard → discard clip.
    Nothing is saved.

Collect mode (LIPSYNC_COLLECT=1):
    After transcription, a review dialog opens. Saving writes the clip,
    metadata, and your corrected transcript to data/manifest.jsonl for
    future fine-tuning. This is the old data-collection workflow, now opt-in.

Optional rescoring (LIPSYNC_RESCORE=1):
    AVSR output is post-processed by rescore.py:polish() (LLM cleanup).
"""
from __future__ import annotations

import queue
import threading
from enum import Enum, auto

from dotenv import load_dotenv
from pynput import keyboard

import avsr
import config
import dataset
from output import deliver
from recorder import CaptureMeta, Recorder
from review import review

HOTKEY = "<cmd>+<shift>+l"


class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


class App:
    def __init__(self) -> None:
        self.state = State.IDLE
        self.recorder: Recorder | None = None  # type: ignore[valid-type]
        self.lock = threading.Lock()
        self.events: queue.Queue = queue.Queue()

    def on_hotkey(self) -> None:
        self.events.put("toggle")

    def run(self) -> None:
        load_dotenv()
        mode = "collect" if config.COLLECT_MODE else "transcribe-only"
        print(f"Backend: {config.AVSR_BACKEND}  |  Mode: {mode}  |  Hotkey: {HOTKEY}")
        if config.COLLECT_MODE:
            print("(Saving clips to data/manifest.jsonl. Set LIPSYNC_COLLECT=0 to disable.)")
        hotkey = keyboard.GlobalHotKeys({HOTKEY: self.on_hotkey})
        hotkey.start()
        try:
            while True:
                self._handle(self.events.get())
        except KeyboardInterrupt:
            print("\nBye.")
        finally:
            hotkey.stop()

    def _handle(self, _event: str) -> None:
        with self.lock:
            if self.state == State.IDLE:
                self._start_recording()
            elif self.state == State.RECORDING:
                self._stop_recording()
            elif self.state == State.PROCESSING:
                print("(still processing previous clip, ignoring)")

    def _start_recording(self) -> None:
        try:
            self.recorder = Recorder()
            path = self.recorder.start(on_complete=self.on_hotkey)
            self.state = State.RECORDING
            print(f"● Recording to {path} (press hotkey again, or 8s auto-stop)")
        except Exception as e:
            print(f"Recorder error: {e}")
            self.recorder = None
            self.state = State.IDLE

    def _stop_recording(self) -> None:
        if not self.recorder:
            self.state = State.IDLE
            return
        meta = self.recorder.stop()
        self.recorder = None
        self.state = State.PROCESSING
        print(f"⏺ Stopped ({meta.duration_seconds}s). Transcribing…")
        threading.Thread(target=self._process, args=(meta,), daemon=True).start()

    def _process(self, meta: CaptureMeta) -> None:
        try:
            try:
                result = avsr.transcribe(meta.path)
            except avsr.NotInstalled as e:
                print(f"\nAVSR backend not installed:\n  {e}\n")
                meta.path.unlink(missing_ok=True)
                return

            text = result.text or ""
            if config.AVSR_RESCORE and text and not text.startswith("["):
                try:
                    from rescore import polish
                    text = polish(text).text
                except Exception as e:
                    print(f"(rescore failed: {e}, keeping raw)")

            if not text:
                print("(empty transcript)")
                meta.path.unlink(missing_ok=True)
                return

            deliver(text)

            if config.COLLECT_MODE:
                self._collect(meta, result, text)
            else:
                meta.path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Inference error: {e}")
            meta.path.unlink(missing_ok=True)
        finally:
            with self.lock:
                self.state = State.IDLE

    def _collect(self, meta: CaptureMeta, result: avsr.AVSRResult, text: str) -> None:
        """Optional: write clip + manifest entry for future fine-tuning."""
        session_id = dataset.new_session_id()
        stored = dataset.import_clip(meta.path, session_id)
        action, corrected = review(text)
        if action == "delete":
            stored.unlink(missing_ok=True)
            dataset.append(
                dataset.Entry(
                    session_id=session_id,
                    speaker_id=config.SPEAKER_ID,
                    pseudo_transcript=text,
                    pseudo_label_model=result.model_id,
                    deleted=True,
                )
            )
            print("(clip discarded)")
            return
        human = corrected if action == "save" else None
        entry = dataset.Entry(
            session_id=session_id,
            speaker_id=config.SPEAKER_ID,
            raw_clip_path=str(stored.relative_to(dataset.DATA_DIR)),
            duration_seconds=meta.duration_seconds,
            fps=meta.fps,
            resolution=meta.resolution,
            audio_present=meta.audio_present,
            language="en",
            pseudo_transcript=text,
            pseudo_label_model=result.model_id,
            prompt_version=None,
            transcript_human=human,
            review_status="reviewed" if action == "save" else "unreviewed",
            label_quality="gold" if action == "save" else "silver",
        )
        dataset.append(entry)
        print(
            f"(saved {session_id}: quality={entry.label_quality}, "
            f"review={entry.review_status})"
        )


if __name__ == "__main__":
    App().run()
