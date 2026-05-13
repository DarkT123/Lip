import subprocess
from typing import Literal

Action = Literal["save", "skip", "delete"]

DELIM = "|||"


def review(text: str) -> tuple[Action, str | None]:
    """Show a correction dialog. Returns (action, corrected_text)."""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    dialog = (
        f'set d to display dialog "Edit the text if Gemini got it wrong, '
        f'then choose an action." default answer "{safe}" '
        f'buttons {{"Delete clip", "Skip review", "Save"}} '
        f'default button "Save" with title "lip-to-text review"'
    )
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e", dialog,
                "-e", f'return (button returned of d) & "{DELIM}" & (text returned of d)',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return "skip", None

    out = result.stdout.strip()
    if DELIM not in out:
        return "skip", None
    button, corrected = out.split(DELIM, 1)
    if button == "Save":
        return "save", corrected
    if button == "Delete clip":
        return "delete", None
    return "skip", None
