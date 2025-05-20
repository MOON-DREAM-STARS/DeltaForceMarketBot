# -*- coding: utf-8 -*-

import easyocr
import numpy as np
import time
import os
import gc  # 添加垃圾回收模块

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    from utils import *
else:
    from backend.utils import *


class BuyBot:
    def __init__(self, ocr_engine="easyocr"):
        self.ocr_engine = ocr_engine.lower()
        if self.ocr_engine == "easyocr":
            self.reader = easyocr.Reader(["en"], gpu=False)
        else:
            raise ValueError("ocr_engine 仅支持 'easyocr'")
        self.range_isconvertible_lowest_price = [
            2179 / 2560,
            1078 / 1440,
            2308 / 2560,
            1102 / 1440,
        ]
        self.range_notconvertible_lowest_price = [
            2179 / 2560,
            1156 / 1440,
            2308 / 2560,
            1178 / 1440,
        ]
        self.postion_isconvertible_max_shopping_number = [0.9085, 0.7222]
        self.postion_isconvertible_min_shopping_number = [0.7921, 0.7222]
        self.postion_notconvertiable_max_shopping_number = [2329 / 2560, 1112 / 1440]
        self.postion_notconvertiable_min_shopping_number = [2028 / 2560, 1112 / 1440]
        self.postion_isconvertible_buy_button = [2189 / 2560, 0.7979]
        self.postion_notconvertiable_buy_button = [2186 / 2560, 1225 / 1440]
        self.lowest_price = None
        print(f"初始化完成，当前OCR引擎: {self.ocr_engine}")

    def detect_price(self, is_convertible, debug_mode=False):
        try:
            # 截图已经返回numpy数组了，不需要再转换
            if is_convertible:
                img_np = get_windowshot(
                    self.range_isconvertible_lowest_price, debug_mode=debug_mode
                )
            else:
                img_np = get_windowshot(
                    self.range_notconvertible_lowest_price, debug_mode=debug_mode
                )

            # 处理OCR识别
            result = self.reader.readtext(img_np)
            if debug_mode:
                print(result)

            # 取最后一个识别结果
            self.lowest_price = int(result[-1][1].replace(",", ""))

            # 清理变量
            del img_np
            del result
            gc.collect()

        except Exception as e:
            self.lowest_price = None
            print(f"识别失败, 建议检查物品是否可兑换，错误信息: {e}")

        return self.lowest_price

    def buy(self, is_convertible, pre_clicked=False):
        """
        执行购买操作
        is_convertible: 是否可兑换
        pre_clicked: 是否已经预先点击过最大购买量
        """
        if is_convertible:
            if not pre_clicked:
                # 只在没有预点击时才点击最大购买量
                mouse_click(self.postion_isconvertible_max_shopping_number)
            # 点击购买按钮
            mouse_click(self.postion_isconvertible_buy_button)
        else:
            # 不可兑换时正常操作
            mouse_click(self.postion_notconvertiable_max_shopping_number)
            mouse_click(self.postion_notconvertiable_buy_button)

    def refresh(self, is_convertible):
        if is_convertible:
            mouse_click(self.postion_isconvertible_min_shopping_number)
            mouse_click(self.postion_isconvertible_buy_button)
        else:
            mouse_click(self.postion_notconvertiable_min_shopping_number)
            mouse_click(self.postion_notconvertiable_buy_button)

    def freerefresh(self, good_postion):
        # esc回到商店页面
        pyautogui.press("esc")
        # 点击回到商品页面
        mouse_click(good_postion)


def main():
    bot = BuyBot(ocr_engine="easyocr")
    is_convertiable = False
    bot.detect_price(is_convertible=is_convertiable, debug_mode=True)


if __name__ == "__main__":
    main()
