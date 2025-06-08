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
    def __init__(self, ocr_engine="easyocr", screenshot_method="mss"):
        self.ocr_engine = ocr_engine.lower()
        self.screenshot_method = screenshot_method.lower()

        if self.ocr_engine == "easyocr":
            self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        else:
            raise ValueError("ocr_engine 仅支持 'easyocr'")

        if self.screenshot_method not in ["mss", "win32"]:
            raise ValueError("screenshot_method 仅支持 'mss' 或 'win32'")

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
        print(
            f"初始化完成，当前OCR引擎: {self.ocr_engine}，截图方法: {self.screenshot_method}"
        )

    def detect_price(self, is_convertible, debug_mode=False):
        try:
            # 使用指定的截图方法
            if is_convertible:
                img_np = get_windowshot(
                    self.range_isconvertible_lowest_price,
                    method=self.screenshot_method,
                    debug_mode=debug_mode,
                )
            else:
                img_np = get_windowshot(
                    self.range_notconvertible_lowest_price,
                    method=self.screenshot_method,
                    debug_mode=debug_mode,
                )

            # 检查截图是否成功
            if img_np is None:
                print(f"{self.screenshot_method}截图失败")
                self.lowest_price = None
                return self.lowest_price

            # 处理OCR识别
            result = self.reader.readtext(img_np)
            if debug_mode:
                print(f"OCR识别结果: {result}")

            # 检查OCR结果是否为空
            if not result:
                print("OCR识别结果为空")
                self.lowest_price = None
                return self.lowest_price

            # 尝试从OCR结果中提取价格
            price_text = None
            for detection in result:
                text = detection[1]
                # 查找包含数字和可能包含逗号/点的文本
                if any(char.isdigit() for char in text):
                    price_text = text
                    break

            if price_text is None:
                print("未在OCR结果中找到有效的价格文本")
                self.lowest_price = None
                return self.lowest_price

            if debug_mode:
                print(f"提取到的价格文本: '{price_text}'")

            # 智能清理价格文本并转换为整数
            cleaned_price = self.parse_price_text(price_text)

            if cleaned_price is not None:
                self.lowest_price = cleaned_price
            else:
                print(f"无法解析价格文本: '{price_text}'")
                self.lowest_price = None

            # 清理变量
            del img_np
            del result
            gc.collect()

        except Exception as e:
            self.lowest_price = None
            print(f"识别失败, 建议检查物品是否可兑换，错误信息: {e}")
            # 保存调试图片
            try:
                if is_convertible:
                    img_np = get_windowshot(
                        self.range_isconvertible_lowest_price,
                        method=self.screenshot_method,
                        debug_mode=True,
                    )
                else:
                    img_np = get_windowshot(
                        self.range_notconvertible_lowest_price,
                        method=self.screenshot_method,
                        debug_mode=True,
                    )
                print("已保存调试截图，请检查screenshot_xxx.png文件")
            except:
                print("无法保存调试截图")

        return self.lowest_price

    def parse_price_text(self, price_text):
        """
        智能解析价格文本，处理各种OCR识别错误
        """
        import re

        if not price_text:
            return None

        # 去除所有空格
        text = price_text.replace(" ", "")

        print(f"正在解析价格文本: '{text}'")

        # 情况1: 正常的逗号分隔千分位，如 "4,177" 或 "1,234,567"
        if "," in text and "." not in text:
            try:
                # 直接移除逗号
                cleaned = text.replace(",", "")
                if cleaned.isdigit():
                    result = int(cleaned)
                    print(f"解析为逗号分隔格式: {result}")
                    return result
            except:
                pass

        # 情况2: OCR把逗号识别成了点，如 "4.177" 或 "4177.452"
        if "." in text:
            # 检查是否像千分位分隔符（没有小数部分或小数部分是3位数字）
            parts = text.split(".")
            if len(parts) == 2:
                integer_part, decimal_part = parts
                # 如果"小数部分"是3位数字，很可能是千分位
                if (
                    len(decimal_part) == 3
                    and decimal_part.isdigit()
                    and integer_part.isdigit()
                ):
                    try:
                        result = int(integer_part + decimal_part)
                        print(f"解析为错误识别的千分位格式: {result}")
                        return result
                    except:
                        pass
                # 如果"小数部分"不是标准的2位小数，也可能是千分位
                elif (
                    len(decimal_part) != 2
                    and decimal_part.isdigit()
                    and integer_part.isdigit()
                ):
                    try:
                        result = int(integer_part + decimal_part)
                        print(f"解析为可能的千分位格式: {result}")
                        return result
                    except:
                        pass

            # 处理多个点的情况，如 "1.234.567"
            if len(parts) > 2:
                try:
                    # 假设都是千分位分隔符
                    cleaned = text.replace(".", "")
                    if cleaned.isdigit():
                        result = int(cleaned)
                        print(f"解析为多点千分位格式: {result}")
                        return result
                except:
                    pass

        # 情况3: 只有数字，没有分隔符
        if text.isdigit():
            try:
                result = int(text)
                print(f"解析为纯数字格式: {result}")
                return result
            except:
                pass

        # 情况4: 使用正则表达式提取所有数字，然后组合
        numbers = re.findall(r"\d+", text)
        if numbers:
            try:
                # 将所有数字部分连接起来
                combined = "".join(numbers)
                if combined:
                    result = int(combined)
                    print(f"解析为正则提取格式: {result}")
                    return result
            except:
                pass

        # 情况5: 处理真正的小数，取整数部分
        try:
            float_value = float(text.replace(",", ""))
            result = int(float_value)
            print(f"解析为浮点数格式: {result}")
            return result
        except:
            pass

        print(f"无法解析价格文本: '{text}'")
        return None

    def buy(self, is_convertible, pre_clicked=True):
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
    # 测试两种截图方法
    print("测试MSS方法...")
    bot_mss = BuyBot(ocr_engine="easyocr", screenshot_method="mss")
    bot_mss.detect_price(is_convertible=False, debug_mode=True)

    print("\n测试Win32方法...")
    bot_win32 = BuyBot(ocr_engine="easyocr", screenshot_method="win32")
    bot_win32.detect_price(is_convertible=False, debug_mode=True)


if __name__ == "__main__":
    main()
