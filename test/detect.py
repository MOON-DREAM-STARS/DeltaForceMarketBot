import os
import time
import csv
import re
import pyautogui
from PIL import Image
import easyocr
import numpy as np
import keyboard
import threading
from concurrent.futures import ThreadPoolExecutor
import gc  # 添加垃圾回收

# 导入优化的鼠标点击函数
from backend.utils import mouse_click

# TODO: 指定"总裁室会客厅"按钮中心坐标
BUTTON_POS = (440, 220)
# TODO: 用 displayMousePosition 校准后，填入正确的 (left, top, width, height)
REGION = (430, 180, 1830, 820)
OUT_CSV = "prices.csv"

# 小矩形区域配置 (right-bottom coordinates: right, bottom, width, height)
RECT_WIDTH = 170
RECT_HEIGHT = 25

# 按行组织的区域配置 (4行，每行3个区域)
ROWS = [
    # 第一行 (3个区域)
    [
        (890, 330, RECT_WIDTH, RECT_HEIGHT),  # 第1列
        (1360, 330, RECT_WIDTH, RECT_HEIGHT),  # 第2列
        (1825, 330, RECT_WIDTH, RECT_HEIGHT),  # 第3列
    ],
    # 第二行 (3个区域)
    [
        (890, 490, RECT_WIDTH, RECT_HEIGHT),  # 第1列
        (1360, 490, RECT_WIDTH, RECT_HEIGHT),  # 第2列
        (1825, 490, RECT_WIDTH, RECT_HEIGHT),  # 第3列
    ],
    # 第三行 (3个区域)
    [
        (890, 650, RECT_WIDTH, RECT_HEIGHT),  # 第1列
        (1360, 650, RECT_WIDTH, RECT_HEIGHT),  # 第2列
        (1825, 650, RECT_WIDTH, RECT_HEIGHT),  # 第3列
    ],
    # 第四行 (3个区域)
    [
        (890, 810, RECT_WIDTH, RECT_HEIGHT),  # 第1列
        (1360, 810, RECT_WIDTH, RECT_HEIGHT),  # 第2列
        (1825, 810, RECT_WIDTH, RECT_HEIGHT),  # 第3列
    ],
]

# 预计算所有区域的左上角坐标，避免重复计算
PROCESSED_ROWS = []
for row in ROWS:
    processed_row = []
    for right, bottom, width, height in row:
        left = right - width
        top = bottom - height
        processed_row.append((left, top, width, height))
    PROCESSED_ROWS.append(processed_row)

# 初始化四个 EasyOCR 识别器（每行一个）
print("正在初始化 4 个 OCR 识别器...")
ocr_readers = []
for i in range(4):
    print(f"初始化 OCR 识别器 {i+1}/4...")
    reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
    ocr_readers.append(reader)
print("所有 OCR 识别器初始化完成!")

# 控制变量
is_running = False
should_exit = False

# 字符修正映射表 - 优化为更快的查找
CHAR_FIXES = {
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
    # 分隔符修正 - 统一转换为逗号
    "，": ",",
    "。": ",",
    "、": ",",
    ".": ",",
    "；": ",",
    ";": ",",
    "：": ",",
    ":": ",",
    " ": "",  # 移除空格
}

# 预编译正则表达式
DIGIT_PATTERN = re.compile(r"\d")
NON_DIGIT_COMMA_PATTERN = re.compile(r"[^\d,]")
NON_DIGIT_PATTERN = re.compile(r"[^\d]")
DIGIT_COMMA_PATTERN = re.compile(r"^[\d,]+$")


def fix_ocr_text(text):
    """修正OCR识别错误的字符 - 优化版本"""
    if not text:
        return text

    # 使用列表推导式和join，比字符串连接更快
    return "".join(CHAR_FIXES.get(char, char) for char in text)


def extract_complete_price(text):
    """从文本中提取完整的价格数字 - 优化版本"""
    if not text:
        return None

    # 先修正OCR错误
    fixed_text = fix_ocr_text(text)

    # 移除所有非数字非逗号字符
    clean_text = NON_DIGIT_COMMA_PATTERN.sub("", fixed_text)

    # 如果没有数字，返回None
    if not DIGIT_PATTERN.search(clean_text):
        return None

    # 提取所有数字
    all_digits = NON_DIGIT_PATTERN.sub("", clean_text)

    # 检查数字长度
    if len(all_digits) < 3:
        return None

    # 检查价格范围
    try:
        price = int(all_digits)
        if 100 <= price <= 99999999:
            return price
    except ValueError:
        pass

    return None


def is_price_format(text):
    """判断文本是否可能包含价格 - 优化版本"""
    if not text or len(text) < 3:
        return False

    # 使用预编译的正则表达式
    digits = DIGIT_PATTERN.findall(text)
    if len(digits) < 3:
        return False

    # 数字占比检查
    return len(digits) / len(text) >= 0.5


def extract_price_from_region(img: Image.Image, reader):
    """从单个小区域提取价格 - 优化版本"""
    try:
        # 将 PIL Image 转为 numpy 数组
        arr = np.array(img)
        result = reader.readtext(arr, detail=0, paragraph=False)

        if not result:
            return None

        # 优化：先尝试最长的文本（通常是完整价格）
        result_sorted = sorted(result, key=len, reverse=True)

        for txt in result_sorted:
            if is_price_format(txt):
                price = extract_complete_price(txt)
                if price is not None:
                    return price

        # 如果单个文本都不行，尝试组合
        combined_text = "".join(result)
        return extract_complete_price(combined_text)

    except Exception:
        return None


def process_row(row_index, row_regions):
    """处理单行的所有区域 - 优化版本"""
    reader = ocr_readers[row_index]
    row_prices = []

    for col_index, lt_region in enumerate(row_regions):
        try:
            # 直接使用预处理的坐标
            small_img = pyautogui.screenshot(region=lt_region)
            price = extract_price_from_region(small_img, reader)

            if price is not None:
                row_prices.append(price)
                print(f"行{row_index+1}-列{col_index+1}: {price}")
            else:
                row_prices.append(0)
                print(f"行{row_index+1}-列{col_index+1}: 未识别到价格")

            # 清理图像对象
            del small_img

        except Exception as e:
            print(f"行{row_index+1}-列{col_index+1}识别出错: {e}")
            row_prices.append(0)

    return row_index, row_prices


def extract_all_prices():
    """并行处理四行区域并提取价格 - 优化版本"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 直接使用预处理的坐标
        futures = [
            executor.submit(process_row, row_idx, PROCESSED_ROWS[row_idx])
            for row_idx in range(4)
        ]

        # 收集结果
        all_results = [future.result() for future in futures]

    # 按行索引排序并提取数据
    all_results.sort(key=lambda x: x[0])
    rows_data = [row_prices for _, row_prices in all_results]

    return rows_data


def monitor_prices():
    global is_running, should_exit

    # 预先打开CSV文件句柄，避免重复打开关闭
    csv_file = None
    csv_writer = None

    if not os.path.exists(OUT_CSV):
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["timestamp"]
            for row in range(1, 5):
                for col in range(1, 4):
                    header.append(f"R{row}C{col}")
            writer.writerow(header)

    try:
        csv_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

        while not should_exit:
            if is_running:
                # 使用优化的鼠标点击函数
                mouse_click(BUTTON_POS)
                pyautogui.press("esc")
                pyautogui.moveTo(100, 900)

                # 获取当前时间戳
                ts = time.strftime("%Y-%m-%d %H:%M:%S")

                # 并行提取所有区域的价格
                start_time = time.time()
                rows_data = extract_all_prices()
                end_time = time.time()

                # 展平数据
                flattened_prices = [
                    price for row_prices in rows_data for price in row_prices
                ]

                # 写入CSV（使用已打开的文件句柄）
                csv_writer.writerow([ts] + flattened_prices)
                csv_file.flush()  # 确保数据写入

                # 统计和显示结果
                valid_prices = [p for p in flattened_prices if p > 0]
                print(f"{ts} -> 共识别到 {len(valid_prices)}/12 个有效价格")

                for i, row_prices in enumerate(rows_data):
                    print(f"第{i+1}行: {row_prices}")

                print(f"处理时间: {end_time - start_time:.2f} 秒")
                print("-" * 50)

                # 定期清理内存
                if time.time() % 60 < 1:  # 大约每分钟清理一次
                    gc.collect()

            else:
                time.sleep(0.1)

    finally:
        if csv_file:
            csv_file.close()


def toggle_running():
    global is_running
    is_running = not is_running
    print("开始监控价格..." if is_running else "停止监控价格...")


def main():
    global should_exit

    print("程序已启动，按 F7 开始/停止监控，按 Ctrl+C 退出程序")
    print(f"配置了 4 行，每行 3 个区域，共 {sum(len(row) for row in ROWS)} 个识别区域")
    print("使用 4 个并行OCR处理器，每个处理器负责一行")

    keyboard.add_hotkey("f7", toggle_running)

    monitor_thread = threading.Thread(target=monitor_prices, daemon=True)
    monitor_thread.start()

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        should_exit = True
        print("\n程序正在退出...")


if __name__ == "__main__":
    main()
