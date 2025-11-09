import streamlit as st
import streamlit_hotkeys as hotkeys

SHORTCUTS = {
    "palette": [
        {
            "key": "k",
            "ctrl": True,
            "prevent_default": True,
            "help": "Open command palette",
        },
    ],
    "save": [
        {
            "key": "s",
            "ctrl": True,
            "prevent_default": True,
            "help": "Save Poll configuration",
        },
    ],
    "rank": [
        {
            "key": "r",
            "ctrl": True,
            "prevent_default": True,
            "help": "Rank completions",
        }
    ],
    "run": [
        {
            "key": "Enter",
            "ctrl": True,
            "prevent_default": True,
            "help": "Run Poll task",
        }
    ],
}


@st.dialog("Keyboard Shortcuts")
def _shortcuts():
    hotkeys.legend()


def activate_shortcuts(fn_map: dict[str, callable]) -> None:
    hotkeys.activate(
        SHORTCUTS,
        key="global",
    )

    if hotkeys.pressed("palette"):
        _shortcuts()

    for name, fn in fn_map.items():
        if hotkeys.pressed(name):
            fn()
