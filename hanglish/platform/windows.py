"""Windows implementation of the base hook interface for Hanglish IME."""
from typing import Tuple

from .base import BaseHook
from ..core.romanizer import Romanizer

class WinHook(BaseHook):
    def __init__(self):
        self.romanizer = Romanizer()
        self.buffer = ""
        
    def _on_press(self, key):
        pass
        
    def _commit_buffer(self, append_space: bool = False, append_enter: bool = False) -> None:
        pass
        
    def commit_text(self, text: str) -> None:
        pass

    def get_cursor_position(self) -> Tuple[int, int]:
        return (0, 0)
        
    def start(self) -> None:
        print("Starting Windows Keyboard Hook... (Not fully implemented)")
