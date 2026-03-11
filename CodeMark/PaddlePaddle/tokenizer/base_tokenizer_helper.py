from typing import List, Tuple


class BaseTokenizerHelper:
    def __init__(self):
        pass

    def encode(self, text: str) -> Tuple[List[int], int]:
        raise NotImplemented

    def decode(self):
        raise NotImplemented

    def get_pad_id(self):
        raise NotImplemented
