import keyboard
import pyautogui
import time
from PIL import Image
import numpy as np
import easyocr
import re


def extract_region_and_ocr(x1, y1, x2, y2, filename_prefix="region"):
    """提取指定矩形区域并进行OCR识别"""
    try:
        # 截取整个屏幕
        screenshot = pyautogui.screenshot()

        # 提取指定区域 (x1,y1) 到 (x2,y2)
        region = screenshot.crop((x1, y1, x2, y2))

        # 保存区域图片
        region_filename = f"{filename_prefix}_region_{x1}_{y1}_{x2}_{y2}.png"
        region.save(region_filename)
        print(f"区域图片已保存: {region_filename}")

        # 转换为numpy数组进行OCR
        region_np = np.array(region)

        # 使用EasyOCR识别数字
        reader = easyocr.Reader(["en"], gpu=False)
        results = reader.readtext(region_np)

        print(f"OCR识别结果: {results}")

        # 提取数字
        numbers = []
        for detection in results:
            text = detection[1]
            # 使用正则表达式提取数字
            digits = re.findall(r"\d+", text)
            numbers.extend(digits)

        if numbers:
            print(f"识别到的数字: {numbers}")
            # 如果需要合并所有数字
            combined_number = "".join(numbers)
            print(f"合并后的数字: {combined_number}")
            return combined_number
        else:
            print("未识别到数字")
            return None

    except Exception as e:
        print(f"OCR识别失败: {e}")
        return None


def main():
    print("程序已启动，按 F7 开始操作...")
    print("操作流程：")
    print("1. 按 F7 开始")
    print("2. 鼠标移动到 (1690, 50)")
    print("3. 提取区域 (1580,270) 到 (1680,290) 并识别数字")
    print("按 ESC 退出程序")

    while True:
        # 检测F7按键
        if keyboard.is_pressed("f7"):
            print("检测到 F7 按键，开始执行操作...")

            # 移动鼠标到指定位置
            print("正在移动鼠标到 (1690, 50)...")
            pyautogui.moveTo(1690, 50)

            # 等待移动完成并稳定
            time.sleep(0.5)
            print("鼠标已移动到位，开始提取区域...")

            # 提取指定矩形区域并识别数字
            result = extract_region_and_ocr(1580, 270, 1680, 292)

            if result:
                print(f"最终识别结果: {result}")

            print("操作完成！等待下次 F7 按键...")

            # 等待 F7 释放，避免重复触发
            while keyboard.is_pressed("f7"):
                time.sleep(0.1)

        # 检测 ESC 退出
        if keyboard.is_pressed("esc"):
            print("检测到 ESC，程序退出")
            break

        # 短暂休眠，避免CPU占用过高
        time.sleep(0.1)


if __name__ == "__main__":
    main()
