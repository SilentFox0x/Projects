from typing import List, Tuple
import random

watermark_len = 0


def watermark_class_to_watermark(watermark_class: int) -> List[int]:
    w_bits = [int(item) for item in list(f'{watermark_class:b}')]
    while len(w_bits) < watermark_len:
        w_bits.insert(0, 0)
    return w_bits


class WatermarkFactory:
    def __init__(self, _watermark_len: int, fix_w_bits: List[int] = None):
        global watermark_len
        watermark_len = _watermark_len
        self.w_candidates = []
        for watermark_class in range(1 << watermark_len):
            w_bits = watermark_class_to_watermark(watermark_class)
            self.w_candidates.append(w_bits)
        if fix_w_bits is not None:
            self.fix_w_bits = fix_w_bits
            bit_to_s = ''.join([str(item) for item in fix_w_bits])
            self.fix_w_bits_class = int(bit_to_s, 2)
        else:
            self.fix_w_bits = None
            self.fix_w_bits_class = None

    def random_watermark(self) -> Tuple[List[int], int]:
        if self.fix_w_bits is not None:
            return self.fix_w_bits, self.fix_w_bits_class
        else:
            w_bits_class = random.randint(0, (1 << watermark_len) - 1)
            w_bits = self.w_candidates[w_bits_class]
            return w_bits, w_bits_class
