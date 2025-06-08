import easyocr
import numpy as np
import time
import os
import gc  # 添加垃圾回收模块
import re

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

        # 预编译正则表达式，避免重复编译
        self._digit_pattern = re.compile(r"\d+")

        # 缓存字符修正映射，避免重复创建字典
        self._char_fixes = {
            # 数字0的常见误识别
            "o": "0",
            "O": "0",
            "g": "0",
            "q": "0",
            "Q": "0",
            # 数字1的常见误识别
            "l": "1",
            "I": "1",
            "|": "1",
            # 数字5的常见误识别
            "S": "5",
            "s": "5",
            # 数字6的常见误识别
            "G": "6",
            "b": "6",
            # 数字8的常见误识别
            "B": "8",
            # 数字2的常见误识别
            "Z": "2",
            "z": "2",
        }

        print(
            f"初始化完成，当前OCR引擎: {self.ocr_engine}，截图方法: {self.screenshot_method}"
        )

    def detect_price(self, is_convertible, debug_mode=False):
        try:
            # 使用指定的截图方法
            screenshot_range = (
                self.range_isconvertible_lowest_price
                if is_convertible
                else self.range_notconvertible_lowest_price
            )

            img_np = get_windowshot(
                screenshot_range,
                method=self.screenshot_method,
                debug_mode=debug_mode,
            )

            # 检查截图是否成功
            if img_np is None:
                print(f"{self.screenshot_method}截图失败")
                self.lowest_price = None
                return self.lowest_price

            # 优化OCR处理 - 直接处理结果，避免重复变量赋值
            ocr_results = self.reader.readtext(img_np)
            if debug_mode:
                print(f"OCR识别结果: {ocr_results}")

            # 检查OCR结果是否为空
            if not ocr_results:
                print("OCR识别结果为空")
                self.lowest_price = None
                return self.lowest_price

            # 优化价格提取 - 使用生成器和next()提前退出
            price_text = next(
                (
                    detection[1]
                    for detection in ocr_results
                    if any(char.isdigit() for char in detection[1])
                ),
                None,
            )

            if price_text is None:
                print("未在OCR结果中找到有效的价格文本")
                self.lowest_price = None
                return self.lowest_price

            if debug_mode:
                print(f"提取到的价格文本: '{price_text}'")

            # 智能清理价格文本并转换为整数
            self.lowest_price = self.parse_price_text(price_text)

            if self.lowest_price is None:
                print(f"无法解析价格文本: '{price_text}'")

            # 清理变量
            del img_np, ocr_results
            gc.collect()

        except Exception as e:
            self.lowest_price = None
            print(f"识别失败, 建议检查物品是否可兑换，错误信息: {e}")
            # 保存调试图片
            try:
                screenshot_range = (
                    self.range_isconvertible_lowest_price
                    if is_convertible
                    else self.range_notconvertible_lowest_price
                )
                img_np = get_windowshot(
                    screenshot_range,
                    method=self.screenshot_method,
                    debug_mode=True,
                )
                print("已保存调试截图，请检查screenshot_xxx.png文件")
            except:
                print("无法保存调试截图")

        return self.lowest_price

    def parse_price_text(self, price_text):
        """
        优化版价格文本解析，减少重复操作
        """
        if not price_text:
            return None

        # 一次性去除空格并修正字符
        text = self.fix_ocr_confusion(price_text.replace(" ", ""))
        print(f"正在解析价格文本: '{text}'")

        # 优化条件判断顺序，最常见情况放前面
        # 情况1: 纯数字（最常见）
        if text.isdigit():
            result = int(text)
            print(f"解析为纯数字格式: {result}")
            return result

        # 情况2: 逗号分隔千分位
        if "," in text and "." not in text:
            cleaned = text.replace(",", "")
            if cleaned.isdigit():
                result = int(cleaned)
                print(f"解析为逗号分隔格式: {result}")
                return result

        # 情况3: 点号替代逗号的情况
        if "." in text:
            parts = text.split(".")
            if len(parts) == 2:
                integer_part, decimal_part = parts
                if (
                    len(decimal_part) == 3
                    and decimal_part.isdigit()
                    and integer_part.isdigit()
                ):
                    result = int(integer_part + decimal_part)
                    print(f"解析为错误识别的千分位格式: {result}")
                    return result
                elif (
                    len(decimal_part) != 2
                    and decimal_part.isdigit()
                    and integer_part.isdigit()
                ):
                    result = int(integer_part + decimal_part)
                    print(f"解析为可能的千分位格式: {result}")
                    return result

            # 多点情况
            if len(parts) > 2:
                cleaned = text.replace(".", "")
                if cleaned.isdigit():
                    result = int(cleaned)
                    print(f"解析为多点千分位格式: {result}")
                    return result

        # 情况4: 正则提取（最后尝试）
        numbers = self._digit_pattern.findall(text)
        if numbers:
            combined = "".join(numbers)
            if combined:
                result = int(combined)
                print(f"解析为正则提取格式: {result}")
                return result

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
        优化版字符修正，减少字符串操作
        """
        if not text:
            return text

        # 使用列表推导式和join，比字符串拼接更高效
        chars = list(text)
        digit_count = sum(1 for c in text if c.isdigit())
        total_chars = len(text.replace(" ", "").replace(",", "").replace(".", ""))
        is_mostly_numbers = digit_count / max(total_chars, 1) > 0.6

        for i, char in enumerate(chars):
            if char in self._char_fixes:
                context_left = chars[i - 1] if i > 0 else ""
                context_right = chars[i + 1] if i < len(chars) - 1 else ""

                is_number_context = (
                    (context_left.isdigit() or context_left in ",.")
                    and (context_right.isdigit() or context_right in ",.")
                ) or (
                    (i == 0 and context_right.isdigit())
                    or (i == len(chars) - 1 and context_left.isdigit())
                )

                if is_number_context or is_mostly_numbers:
                    # 特殊处理点号：只有在不是小数点的情况下才替换为逗号
                    if char == "." and self._char_fixes.get(char) == ",":
                        if i + 3 < len(chars):
                            right_part = "".join(chars[i + 1 : i + 4])
                            if right_part.isdigit():
                                chars[i] = ","
                    else:
                        chars[i] = self._char_fixes[char]

        return "".join(chars)

    def buy(self, is_convertible):
        """
        执行购买操作
        is_convertible: 是否可兑换
        """
        if is_convertible:
            # 点击最大购买量
            mouse_click(self.postion_isconvertible_max_shopping_number)
            # 点击购买按钮
            mouse_click(self.postion_isconvertible_buy_button)
        else:
            # 不可兑换时正常操作
            mouse_click(self.postion_notconvertiable_max_shopping_number)
            mouse_click(self.postion_notconvertiable_buy_button)

    def refresh(self, is_convertible):
        positions = (
            (
                self.postion_isconvertible_min_shopping_number,
                self.postion_isconvertible_buy_button,
            )
            if is_convertible
            else (
                self.postion_notconvertiable_min_shopping_number,
                self.postion_notconvertiable_buy_button,
            )
        )
        mouse_click(positions[0])
        mouse_click(positions[1])

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
