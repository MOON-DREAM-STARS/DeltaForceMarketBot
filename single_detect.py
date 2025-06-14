import os
import time
import csv
import re
import pyautogui
from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import numpy as np
import keyboard
import threading
import gc
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 导入优化的鼠标点击函数
from backend.utils import mouse_click

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 配置常量 ====================
BUTTON_POS = (440, 220)
REGION = (430, 180, 1830, 820)
OUT_CSV = "single_price.csv"

# 详情界面价格区域 (right, bottom, width, height)
PRICE_REGIONS_RB = [
    (268, 953, 106, 19),  # 最低价
    (421, 953, 106, 19),  # 次低价
    (574, 953, 106, 19),  # 第三低价
]

# 详情界面出售数量区域 (right, bottom, width, height)
QUANTITY_REGIONS_RB = [
    (281, 927, 151, 685),  # 第一个价格区域的出售数量
    (434, 927, 151, 685),  # 第二个价格区域的出售数量
    (587, 927, 151, 685),  # 第三个价格区域的出售数量
]

# 转换为左上角坐标格式
PRICE_REGIONS = [
    (right - width, bottom - height, width, height)
    for right, bottom, width, height in PRICE_REGIONS_RB
]

QUANTITY_REGIONS = [
    (right - width, bottom - height, width, height)
    for right, bottom, width, height in QUANTITY_REGIONS_RB
]

# 区域配置
RECT_WIDTH, RECT_HEIGHT = 170, 25
ROWS = [
    [
        (890, 330, RECT_WIDTH, RECT_HEIGHT),
        (1360, 330, RECT_WIDTH, RECT_HEIGHT),
        (1825, 330, RECT_WIDTH, RECT_HEIGHT),
    ],
    [
        (890, 490, RECT_WIDTH, RECT_HEIGHT),
        (1360, 490, RECT_WIDTH, RECT_HEIGHT),
        (1825, 490, RECT_WIDTH, RECT_HEIGHT),
    ],
    [
        (890, 650, RECT_WIDTH, RECT_HEIGHT),
        (1360, 650, RECT_WIDTH, RECT_HEIGHT),
        (1825, 650, RECT_WIDTH, RECT_HEIGHT),
    ],
    [
        (890, 810, RECT_WIDTH, RECT_HEIGHT),
        (1360, 810, RECT_WIDTH, RECT_HEIGHT),
        (1825, 810, RECT_WIDTH, RECT_HEIGHT),
    ],
]

# 监控区域配置
TARGET_ROW, TARGET_COL = 0, 0
if not (0 <= TARGET_ROW <= 3 and 0 <= TARGET_COL <= 2):
    raise ValueError(f"无效的区域配置: ROW={TARGET_ROW}, COL={TARGET_COL}")

TARGET_REGION = ROWS[TARGET_ROW][TARGET_COL]
REGION_ID = TARGET_ROW * 3 + TARGET_COL + 1
REGION_NAME = f"R{TARGET_ROW+1}C{TARGET_COL+1}"


# ==================== OCR 优化配置 ====================
class OCRProcessor:
    """OCR处理器，优化内存使用和处理效率"""

    def __init__(self, processor_id):
        self.processor_id = processor_id
        logger.info(f"初始化 OCR 识别器 #{processor_id}...")
        self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        logger.info(f"OCR 识别器 #{processor_id} 初始化完成!")

        # 字符修正映射表
        self.char_fixes = {
            "o": "0",
            "O": "0",
            "g": "0",
            "q": "0",
            "Q": "0",
            "l": "1",
            "I": "1",
            "|": "1",
            "S": "5",
            "s": "5",
            "G": "6",
            "b": "6",
            "B": "8",
            "Z": "2",
            "z": "2",
            "，": ",",
            "。": ",",
            "、": ",",
            ".": ",",
            "；": ",",
            ";": ",",
            "：": ",",
            ":": ",",
            " ": "",
        }

        # 预编译正则表达式
        self.digit_pattern = re.compile(r"\d")
        self.non_digit_comma_pattern = re.compile(r"[^\d,]")
        self.non_digit_pattern = re.compile(r"[^\d]")

    def fix_text(self, text):
        """修正OCR识别错误的字符"""
        if not text:
            return text
        return "".join(self.char_fixes.get(char, char) for char in text)

    def extract_price(self, text):
        """从文本中提取价格数字"""
        if not text:
            return None

        fixed_text = self.fix_text(text)
        clean_text = self.non_digit_comma_pattern.sub("", fixed_text)

        if not self.digit_pattern.search(clean_text):
            return None

        all_digits = self.non_digit_pattern.sub("", clean_text)
        if len(all_digits) < 3:
            return None

        try:
            price = int(all_digits)
            return price if 100 <= price <= 99999999 else None
        except ValueError:
            return None

    def extract_quantity(self, text):
        """从文本中提取数量数字（与价格类似但范围不同）"""
        if not text:
            return None

        fixed_text = self.fix_text(text)
        clean_text = self.non_digit_comma_pattern.sub("", fixed_text)

        if not self.digit_pattern.search(clean_text):
            return None

        all_digits = self.non_digit_pattern.sub("", clean_text)
        if len(all_digits) < 1:
            return None

        try:
            quantity = int(all_digits)
            return quantity if 1 <= quantity <= 999999999 else None
        except ValueError:
            return None

    def process_price_image(self, img):
        """处理价格图像并提取价格"""
        try:
            arr = np.array(img)
            result = self.reader.readtext(arr, detail=0, paragraph=False)

            if not result:
                return 0

            # 直接处理所有识别结果，优先处理最长文本
            for txt in sorted(result, key=len, reverse=True):
                price = self.extract_price(txt)
                if price is not None:
                    return price

            # 尝试组合文本
            combined = "".join(result)
            return self.extract_price(combined) or 0

        except Exception as e:
            logger.error(f"OCR#{self.processor_id} 价格处理错误: {e}")
            return 0

    def process_quantity_image(self, img, region_index):
        """处理数量图像并提取数量（不使用预处理）"""
        try:
            print(
                f"[DEBUG] OCR#{self.processor_id} 数量区域{region_index+1} 开始处理..."
            )

            arr = np.array(img)
            result = self.reader.readtext(arr, detail=0, paragraph=False)

            print(f"[DEBUG] OCR#{self.processor_id} 识别结果: {result}")

            if result:
                # 直接处理所有识别结果，优先处理最长文本
                for txt in sorted(result, key=len, reverse=True):
                    quantity = self.extract_quantity(txt)
                    if quantity is not None:
                        print(
                            f"[DEBUG] OCR#{self.processor_id} 成功提取数量: {quantity}"
                        )
                        return quantity

                # 尝试组合文本
                combined = "".join(result)
                quantity = self.extract_quantity(combined)
                if quantity is not None:
                    print(
                        f"[DEBUG] OCR#{self.processor_id} 组合文本提取数量: {quantity}"
                    )
                    return quantity

            print(f"[DEBUG] OCR#{self.processor_id} 未识别到有效数量")
            return 0

        except Exception as e:
            logger.error(f"OCR#{self.processor_id} 数量处理错误: {e}")
            return 0


# ==================== 全局变量和初始化 ====================
# 创建三个独立的OCR处理器
ocr_processors = [OCRProcessor(i + 1) for i in range(3)]
is_running = False
should_exit = False

# 使用更大的线程池支持并行OCR处理
executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="OCR-Worker")
result_queue = Queue()


# ==================== 核心功能函数 ====================
def process_single_region(processor, price_img, quantity_img, region_index):
    """处理单个区域的价格和数量"""
    try:
        price = processor.process_price_image(price_img)
        quantity = processor.process_quantity_image(quantity_img, region_index)

        # 立即释放图像内存
        price_img.close()
        quantity_img.close()

        return region_index, price, quantity

    except Exception as e:
        logger.error(f"单区域处理错误 (区域{region_index}): {e}")
        return region_index, 0, 0


def capture_and_process():
    """捕获截图并并行处理OCR"""
    try:
        # 1. 点击进入详情界面
        mouse_click(BUTTON_POS)
        time.sleep(0.15)  # 减少等待时间

        # 2. 快速截图价格和数量区域
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        price_imgs = [pyautogui.screenshot(region=r) for r in PRICE_REGIONS]
        quantity_imgs = [pyautogui.screenshot(region=r) for r in QUANTITY_REGIONS]

        # 保存原始截图
        for idx, img in enumerate(price_imgs):
            img.save(f"debug_price_{idx+1}_original.png")
        for idx, img in enumerate(quantity_imgs):
            img.save(f"debug_quantity_{idx+1}_original.png")

        # 3. 立即退出详情界面
        pyautogui.press("esc")

        # 4. 提交并行OCR任务
        executor.submit(process_ocr_parallel, price_imgs, quantity_imgs, ts)

        return True

    except Exception as e:
        logger.error(f"截图处理错误: {e}")
        return False


def process_ocr_parallel(price_imgs, quantity_imgs, ts):
    """并行处理OCR任务"""
    try:
        # 提交三个并行任务
        futures = []
        for i in range(3):
            future = executor.submit(
                process_single_region,
                ocr_processors[i],
                price_imgs[i],
                quantity_imgs[i],
                i,
            )
            futures.append(future)

        # 等待所有任务完成并收集结果
        results = [None, None, None]  # 预分配结果数组

        for future in as_completed(futures):
            region_index, price, quantity = future.result()
            results[region_index] = (price, quantity)

        # 提取价格和数量列表
        prices = [result[0] for result in results]
        quantities = [result[1] for result in results]

        # 将结果放入队列
        result_queue.put((ts, prices, quantities))

    except Exception as e:
        logger.error(f"并行OCR处理错误: {e}")
        result_queue.put((ts, [0, 0, 0], [0, 0, 0]))


class CSVWriter:
    """优化的CSV写入器"""

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None
        self._init_csv()

    def _init_csv(self):
        """初始化CSV文件"""
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "region",
                        "lowest_price",
                        "second_price",
                        "third_price",
                        "lowest_quantity",
                        "second_quantity",
                        "third_quantity",
                    ]
                )

        self.file = open(self.filename, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

    def write_row(self, data):
        """写入一行数据"""
        try:
            self.writer.writerow(data)
            self.file.flush()
        except Exception as e:
            logger.error(f"CSV写入错误: {e}")

    def close(self):
        """关闭文件"""
        if self.file:
            self.file.close()


def monitor_prices():
    """主监控循环"""
    global is_running, should_exit

    csv_writer = CSVWriter(OUT_CSV)
    last_cleanup = time.time()

    try:
        while not should_exit:
            if is_running:
                # 执行截图和OCR（无需额外等待，OCR时间即为等待时间）
                if capture_and_process():
                    # 处理OCR结果
                    try:
                        ts, prices, quantities = result_queue.get(timeout=5.0)

                        # 写入CSV
                        csv_writer.write_row([ts, REGION_NAME] + prices + quantities)

                        # 打印结果
                        logger.info(f"{ts} -> {REGION_NAME}:")
                        logger.info(
                            f"  价格: 最低={prices[0]}, 次低={prices[1]}, 第三={prices[2]}"
                        )
                        logger.info(
                            f"  数量: 最低={quantities[0]}, 次低={quantities[1]}, 第三={quantities[2]}"
                        )
                        print("-" * 40)

                    except:
                        pass  # 超时或队列为空

                # 定期内存清理 (每30秒)
                current_time = time.time()
                if current_time - last_cleanup > 30:
                    gc.collect()
                    last_cleanup = current_time

            else:
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"监控循环错误: {e}")
    finally:
        csv_writer.close()
        executor.shutdown(wait=False)


# ==================== 控制函数 ====================
def toggle_running():
    global is_running
    is_running = not is_running
    status = "开始监控价格..." if is_running else "停止监控价格..."
    print(status)
    logger.info(status)


def print_region_map():
    """打印区域映射表"""
    print("\n区域映射表:")
    print("=" * 50)
    region_id = 1
    for row_idx, row in enumerate(ROWS):
        row_info = []
        for col_idx, region in enumerate(row):
            status = " ★" if (row_idx == TARGET_ROW and col_idx == TARGET_COL) else ""
            row_info.append(f"R{row_idx+1}C{col_idx+1}(#{region_id}){status}")
            region_id += 1
        print(" | ".join(row_info))
    print("=" * 50)
    print("★ 当前监控区域")


def main():
    global should_exit

    print("=" * 60)
    print("单区域价格监控程序 v3.0 (并行OCR版)")
    print("=" * 60)
    print(f"当前监控区域: {REGION_NAME} (区域#{REGION_ID})")
    print(f"监控内容: 价格 + 出售数量 (3个并行OCR)")
    print(f"按 F7 开始/停止监控，按 Ctrl+C 退出程序")

    print_region_map()

    keyboard.add_hotkey("f7", toggle_running)

    monitor_thread = threading.Thread(target=monitor_prices, daemon=True)
    monitor_thread.start()

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        should_exit = True
        logger.info("程序正在退出...")
        print("\n程序正在退出...")


if __name__ == "__main__":
    main()
