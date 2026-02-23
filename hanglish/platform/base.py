from abc import ABC, abstractmethod
from typing import Tuple

class BaseHook(ABC):
    @abstractmethod
    def start(self) -> None:
        """키보드 입력을 가로채기(Hooking) 시작"""
        pass
    
    @abstractmethod
    def commit_text(self, text: str) -> None:
        """최종 변환된 텍스트를 현재 커서 위치에 삽입"""
        pass
    
    @abstractmethod
    def get_cursor_position(self) -> Tuple[int, int]:
        """현재 텍스트 커서의 (x, y) 화면 좌표 반환 (후보 창 띄우기 위함)"""
        pass
