import subprocess

import pyperclip


def deliver(text: str) -> None:
    pyperclip.copy(text)
    safe = text.replace('"', '\\"')
    subprocess.run(
        ["osascript", "-e", f'display notification "{safe}" with title "lip-to-text"'],
        check=False,
    )
    print(f"→ {text}")
