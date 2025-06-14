import os
import time
import csv
import re
import pyautogui
from PIL import Image, ImageEnhance, ImageFilter
from paddleocr import PaddleOCR
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

# 统一的OCR识别区域：左上角（129，218），右下角（588，979）
# 转换为 (left, top, width, height) 格式
OCR_REGION = (129, 218, 588 - 129, 979 - 218)

# 区域配置（保持原有的网格配置用于定位）
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
class OCRResult:
    """OCR识别结果数据类"""
    def __init__(self, text, confidence, bbox):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox  # 边界框坐标
        self.center_x = (bbox[0][0] + bbox[2][0]) / 2
        self.center_y = (bbox[0][1] + bbox[2][1]) / 2


import cv2

class OCRProcessor:
    """OCR处理器，优化内存使用和处理效率"""

    def __init__(self, processor_id):
        self.processor_id = processor_id
        logger.info(f"初始化 PaddleOCR 识别器 #{processor_id}...")
        # 初始化PaddleOCR
        self.reader = PaddleOCR(use_angle_cls=True, lang="ch")
        logger.info(f"PaddleOCR 识别器 #{processor_id} 初始化完成!")

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

    def enhanced_preprocess_image(self, img):
        """增强版图像预处理"""
        try:
            print(f"[DEBUG] OCR#{self.processor_id} 开始增强图像预处理...")
            
            # 1. 转换为numpy数组
            if isinstance(img, Image.Image):
                # 确保是RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_array = np.array(img)
            else:
                img_array = img
            
            print(f"[DEBUG] OCR#{self.processor_id} 输入图像形状: {img_array.shape}")
            
            # 2. 图像尺寸检查
            if len(img_array.shape) == 2:
                height, width = img_array.shape
                channels = 1
            elif len(img_array.shape) == 3:
                height, width, channels = img_array.shape
            else:
                print(f"[DEBUG] OCR#{self.processor_id} 不支持的图像维度: {img_array.shape}")
                return None
            
            if width < 10 or height < 10:
                print(f"[DEBUG] OCR#{self.processor_id} 图像尺寸过小: {width}x{height}")
                return None
            
            # 3. 转换为OpenCV格式 (BGR)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = img_array
            
            # 4. 转换为灰度图进行增强处理
            if len(img_cv.shape) == 3:
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cv
            
            print(f"[DEBUG] OCR#{self.processor_id} 灰度图像形状: {gray.shape}")
            
            # 5. 图像放大以提高OCR精度
            if width < 500 or height < 200:
                scale_factor = max(500/width, 200/height, 2.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"[DEBUG] OCR#{self.processor_id} 灰度图像放大: {width}x{height} -> {new_width}x{new_height}")
            
            # 6. 直方图均衡化增强对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_gray = clahe.apply(gray)
            
            print(f"[DEBUG] OCR#{self.processor_id} 直方图均衡化完成")
            
            # 7. 将增强后的灰度图转换为3通道RGB格式（PaddleOCR要求）
            img_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            
            print(f"[DEBUG] OCR#{self.processor_id} 最终RGB图像形状: {img_rgb.shape}")
            
            # 8. 保存调试图像
            cv2.imwrite(f"debug_preprocess_original_{self.processor_id}.png", img_cv if len(img_cv.shape) == 3 else cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR))
            cv2.imwrite(f"debug_preprocess_gray_{self.processor_id}.png", gray)
            cv2.imwrite(f"debug_preprocess_enhanced_{self.processor_id}.png", enhanced_gray)
            cv2.imwrite(f"debug_preprocess_final_{self.processor_id}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            
            print(f"[DEBUG] OCR#{self.processor_id} 图像预处理完成，最终形状: {img_rgb.shape}")
            
            return img_rgb
            
        except Exception as e:
            print(f"[DEBUG] OCR#{self.processor_id} 增强图像预处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        

    def safe_ocr_predict(self, img):
        """安全的OCR预测，包含错误处理"""
        try:
            # 图像预处理
            processed_img = self.enhanced_preprocess_image(img)
            if processed_img is None:
                return None
            
            print(f"[DEBUG] OCR#{self.processor_id} 图像形状: {processed_img.shape}")
            
            # 尝试OCR识别
            result = self.reader.predict(processed_img)
            
            # 调试：打印result的完整结构
            print(f"[DEBUG] OCR#{self.processor_id} result类型: {type(result)}")
            print(f"[DEBUG] OCR#{self.processor_id} result长度: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            if result:
                print(f"[DEBUG] OCR#{self.processor_id} result[0]类型: {type(result[0])}")
                if hasattr(result[0], '__len__'):
                    print(f"[DEBUG] OCR#{self.processor_id} result[0]长度: {len(result[0])}")
                print(f"[DEBUG] OCR#{self.processor_id} result结构: {result}")
            
            return result
        
        except Exception as e:
            print(f"[DEBUG] OCR#{self.processor_id} OCR预测失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def parse_ocr_results(self, ocr_result):
        """解析OCR结果，提取文本、置信度和坐标信息"""
        results = []
        
        try:
            if not ocr_result:
                print(f"[DEBUG] OCR#{self.processor_id} ocr_result为空")
                return results
            
            print(f"[DEBUG] OCR#{self.processor_id} 开始解析OCR结果...")
            print(f"[DEBUG] OCR#{self.processor_id} ocr_result类型: {type(ocr_result)}")
            
            # 检查ocr_result的结构
            if not hasattr(ocr_result, '__len__') or len(ocr_result) == 0:
                print(f"[DEBUG] OCR#{self.processor_id} ocr_result无效或为空")
                return results
            
            # 获取第一个元素
            result_data = ocr_result[0]
            print(f"[DEBUG] OCR#{self.processor_id} result_data类型: {type(result_data)}")
            
            # 检查结果结构
            if isinstance(result_data, dict):
                # 字典格式（参考ocr_demo.py）
                print(f"[DEBUG] OCR#{self.processor_id} 处理字典格式结果")
                print(f"[DEBUG] OCR#{self.processor_id} 字典键: {result_data.keys()}")
                
                texts = result_data.get('rec_texts', [])
                scores = result_data.get('rec_scores', [])
                boxes = result_data.get('rec_polys', [])
                
                print(f"[DEBUG] OCR#{self.processor_id} texts数量: {len(texts)}")
                print(f"[DEBUG] OCR#{self.processor_id} scores数量: {len(scores)}")
                print(f"[DEBUG] OCR#{self.processor_id} boxes数量: {len(boxes)}")
                
                for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
                    try:
                        result = OCRResult(text, score, box)
                        results.append(result)
                        print(f"[DEBUG] OCR#{self.processor_id} 识别到[{i}]: '{text}' "
                              f"置信度: {score:.2f} 坐标: ({result.center_x:.1f}, {result.center_y:.1f})")
                    except Exception as e:
                        print(f"[DEBUG] OCR#{self.processor_id} 处理第{i}个结果时出错: {e}")
                        
            elif isinstance(result_data, list):
                # 列表格式（原始格式）
                print(f"[DEBUG] OCR#{self.processor_id} 处理列表格式结果，长度: {len(result_data)}")
                
                for i, line in enumerate(result_data):
                    try:
                        print(f"[DEBUG] OCR#{self.processor_id} 处理第{i}行: {line}")
                        print(f"[DEBUG] OCR#{self.processor_id} line类型: {type(line)}, 长度: {len(line) if hasattr(line, '__len__') else 'N/A'}")
                        
                        if not hasattr(line, '__len__') or len(line) < 2:
                            print(f"[DEBUG] OCR#{self.processor_id} 第{i}行格式无效")
                            continue
                            
                        bbox = line[0]  # 边界框坐标
                        text_info = line[1]  # (文本, 置信度)
                        
                        print(f"[DEBUG] OCR#{self.processor_id} bbox: {bbox}")
                        print(f"[DEBUG] OCR#{self.processor_id} text_info: {text_info}, 类型: {type(text_info)}")
                        
                        if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            result = OCRResult(text, confidence, bbox)
                            results.append(result)
                            
                            print(f"[DEBUG] OCR#{self.processor_id} 识别到[{i}]: '{text}' "
                                  f"置信度: {confidence:.2f} 坐标: ({result.center_x:.1f}, {result.center_y:.1f})")
                        else:
                            print(f"[DEBUG] OCR#{self.processor_id} 第{i}行text_info格式无效")
                            
                    except Exception as e:
                        print(f"[DEBUG] OCR#{self.processor_id} 处理第{i}行时出错: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"[DEBUG] OCR#{self.processor_id} 未知的result_data格式: {type(result_data)}")
            
            print(f"[DEBUG] OCR#{self.processor_id} 解析完成，共识别到{len(results)}个结果")
            return results
            
        except Exception as e:
            print(f"[DEBUG] OCR#{self.processor_id} 解析OCR结果失败: {e}")
            import traceback
            traceback.print_exc()
            return results

    def extract_quantity(self, text):
        """从文本中提取数量"""
        try:
            # 修正字符
            corrected = text
            for old, new in self.char_fixes.items():
                corrected = corrected.replace(old, new)
            
            # 提取数字
            numbers = self.non_digit_pattern.sub("", corrected)
            if numbers and numbers.isdigit():
                return int(numbers)
            return None
        except:
            return None

    def extract_price(self, text):
        """从文本中提取价格"""
        try:
            # 修正字符
            corrected = text
            for old, new in self.char_fixes.items():
                corrected = corrected.replace(old, new)
            
            # 移除非数字和逗号的字符
            cleaned = self.non_digit_comma_pattern.sub("", corrected)
            
            if cleaned:
                # 移除逗号并转换为数字
                price_str = cleaned.replace(",", "")
                if price_str.isdigit():
                    return int(price_str)
            return None
        except:
            return None

    def process_unified_ocr(self, img):
        """处理统一的OCR区域图像，分离数量和价格"""
        try:
            print(f"[DEBUG] OCR#{self.processor_id} 开始处理统一OCR区域...")
            
            # 使用安全的OCR预测
            ocr_result = self.safe_ocr_predict(img)
            
            if ocr_result is None:
                print(f"[DEBUG] OCR#{self.processor_id} OCR预测返回空结果")
                return [0, 0, 0], [0, 0, 0]

            # 解析OCR结果
            parsed_results = self.parse_ocr_results(ocr_result)
            
            if not parsed_results:
                print(f"[DEBUG] OCR#{self.processor_id} 未解析到有效结果")
                return [0, 0, 0], [0, 0, 0]

            # 按Y坐标排序，然后按X坐标排序
            # 数量通常在上方，价格在下方
            parsed_results.sort(key=lambda x: (x.center_y, x.center_x))
            
            print(f"[DEBUG] OCR#{self.processor_id} 排序后的结果:")
            for i, result in enumerate(parsed_results):
                print(f"  [{i}] '{result.text}' 坐标: ({result.center_x:.1f}, {result.center_y:.1f})")
            
            # 分离数量和价格
            quantities = [0, 0, 0]
            prices = [0, 0, 0]
            
            # 根据排序结果，前三个为数量，后三个为价格
            for i, result in enumerate(parsed_results):
                if i < 3:  # 前三个作为数量
                    quantity = self.extract_quantity(result.text)
                    if quantity is not None:
                        quantities[i] = quantity
                        print(f"[DEBUG] OCR#{self.processor_id} 数量{i+1}: {quantity}")
                elif i < 6:  # 后三个作为价格
                    price = self.extract_price(result.text)
                    if price is not None:
                        prices[i-3] = price
                        print(f"[DEBUG] OCR#{self.processor_id} 价格{i-2}: {price}")
            
            print(f"[DEBUG] OCR#{self.processor_id} 最终结果:")
            print(f"  数量: {quantities}")
            print(f"  价格: {prices}")
            
            return quantities, prices

        except Exception as e:
            logger.error(f"OCR#{self.processor_id} 统一OCR处理错误: {str(e)}")
            print(f"[DEBUG] OCR#{self.processor_id} 统一OCR处理异常: {str(e)}")
            return [0, 0, 0], [0, 0, 0]


# ==================== 全局变量和初始化 ====================
# 创建单个OCR处理器
ocr_processor = OCRProcessor(1)
is_running = False
should_exit = False

# 使用较小的线程池
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="OCR-Worker")
result_queue = Queue()


# ==================== 核心功能函数 ====================
def capture_and_process():
    """捕获截图并处理OCR"""
    try:
        # 1. 点击进入详情界面
        mouse_click(BUTTON_POS)
        time.sleep(0.15)  # 减少等待时间

        # 2. 快速截图统一OCR区域
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        ocr_img = pyautogui.screenshot(region=OCR_REGION)

        # 保存调试截图
        ocr_img.save("debug_unified_ocr.png")

        # 3. 立即退出详情界面
        pyautogui.press("esc")

        # 4. 提交OCR任务
        executor.submit(process_ocr_task, ocr_img, ts)

        return True

    except Exception as e:
        logger.error(f"截图处理错误: {e}")
        return False


def process_ocr_task(ocr_img, ts):
    """处理OCR任务"""
    try:
        # 处理统一OCR区域
        quantities, prices = ocr_processor.process_unified_ocr(ocr_img)
        
        # 释放图像内存
        ocr_img.close()

        # 将结果放入队列
        result_queue.put((ts, prices, quantities))

    except Exception as e:
        logger.error(f"OCR处理错误: {e}")
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
                # 执行截图和OCR
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
    print("单区域价格监控程序 v4.1 (PaddleOCR优化版)")
    print("=" * 60)
    print(f"当前监控区域: {REGION_NAME} (区域#{REGION_ID})")
    print(f"OCR区域: 左上角({OCR_REGION[0]}, {OCR_REGION[1]}) 大小({OCR_REGION[2]}x{OCR_REGION[3]})")
    print(f"监控内容: 统一OCR识别，前3个结果为数量，后3个为价格")
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