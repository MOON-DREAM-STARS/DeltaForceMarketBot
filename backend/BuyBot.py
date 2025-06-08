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

        # 首先处理字符混淆问题
        text = self.fix_ocr_confusion(text)
        print(f"字符修正后: '{text}'")

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

    def fix_ocr_confusion(self, text):
        """
        修正OCR常见的字符混淆问题
        """
        if not text:
            return text

        # 创建字符映射表
        char_fixes = {
            # 数字0的常见误识别
            "o": "0",  # 小写字母o -> 数字0
            "O": "0",  # 大写字母O -> 数字0
            "g": "0",  # 字母g -> 数字0 (下半部分可能被识别为g)
            "q": "0",  # 字母q -> 数字0
            "Q": "0",  # 大写字母Q -> 数字0
            # 数字1的常见误识别
            "l": "1",  # 小写字母l -> 数字1
            "I": "1",  # 大写字母I -> 数字1
            "|": "1",  # 竖线 -> 数字1
            # 数字5的常见误识别
            "S": "5",  # 大写字母S -> 数字5
            "s": "5",  # 小写字母s -> 数字5
            # 数字6的常见误识别
            "G": "6",  # 大写字母G -> 数字6
            "b": "6",  # 小写字母b -> 数字6
            # 数字8的常见误识别
            "B": "8",  # 大写字母B -> 数字8
            # 数字2的常见误识别
            "Z": "2",  # 大写字母Z -> 数字2
            "z": "2",  # 小写字母z -> 数字2
            # 千分位分隔符的常见误识别
            ".": ",",  # 点 -> 逗号 (在某些情况下)
        }

        result = text

        # 智能替换：只有在特定上下文中才替换
        for i, char in enumerate(text):
            if char in char_fixes:
                # 检查上下文：如果周围都是数字或分隔符，则进行替换
                context_left = text[i - 1] if i > 0 else ""
                context_right = text[i + 1] if i < len(text) - 1 else ""

                # 如果字符两边都是数字、逗号或点，则很可能是数字识别错误
                is_number_context = (
                    (context_left.isdigit() or context_left in ",.")
                    and (context_right.isdigit() or context_right in ",.")
                ) or (
                    # 或者如果是在开头/结尾且另一边是数字
                    (i == 0 and context_right.isdigit())
                    or (i == len(text) - 1 and context_left.isdigit())
                )

                # 特殊处理：如果整个字符串主要由数字和少量字母组成，则替换
                digit_count = sum(1 for c in text if c.isdigit())
                total_chars = len(
                    text.replace(" ", "").replace(",", "").replace(".", "")
                )
                is_mostly_numbers = digit_count / max(total_chars, 1) > 0.6

                if is_number_context or is_mostly_numbers:
                    # 特殊处理点号：只有在不是小数点的情况下才替换为逗号
                    if char == "." and char_fixes[char] == ",":
                        # 检查是否可能是千分位分隔符
                        if context_right and len(context_right) >= 3:
                            # 检查右边是否有3位数字
                            right_part = text[i + 1 :]
                            if len(right_part) >= 3 and right_part[:3].isdigit():
                                result = result[:i] + "," + result[i + 1 :]
                    else:
                        result = result[:i] + char_fixes[char] + result[i + 1 :]

        return result

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
