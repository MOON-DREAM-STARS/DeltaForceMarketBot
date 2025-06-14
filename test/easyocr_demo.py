import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def ocr_demo(image_path):
    """
    EasyOCR 文本识别演示
    
    Args:
        image_path: 图片路径
    """
    # 初始化 EasyOCR 读取器，支持中文和英文
    reader = easyocr.Reader(['ch_sim', 'en'])
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 进行文本识别
    print("正在识别文本...")
    results = reader.readtext(image)
    
    # 显示识别结果
    print("\n识别结果:")
    print("-" * 50)
    for i, (bbox, text, confidence) in enumerate(results):
        print(f"文本 {i+1}: {text}")
        print(f"置信度: {confidence:.2f}")
        print(f"位置: {bbox}")
        print("-" * 30)
    
    # 在图片上绘制识别结果并显示
    draw_and_show_results(image, results, image_path)

def draw_and_show_results(image, results, image_path):
    """
    在图片上绘制识别结果并显示
    
    Args:
        image: 原始图片
        results: OCR识别结果
        image_path: 图片路径
    """
    # 复制图片用于绘制
    result_image = image.copy()
    
    # 绘制边界框和文本
    for bbox, text, confidence in results:
        # 获取边界框坐标
        points = np.array(bbox, dtype=np.int32)
        
        # 绘制边界框
        cv2.polylines(result_image, [points], True, (0, 255, 0), 2)
        
        # 在边界框上方添加文本
        x, y = points[0]
        cv2.putText(result_image, f"{text} ({confidence:.2f})", 
                   (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 使用OpenCV显示图片
    # 调整窗口大小以适应屏幕
    height, width = result_image.shape[:2]
    if width > 1200 or height > 800:
        scale = min(1200/width, 800/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        result_image = cv2.resize(result_image, (new_width, new_height))
    
    # 显示原图
    cv2.imshow('原始图片', cv2.resize(image, (result_image.shape[1], result_image.shape[0])))
    
    # 显示识别结果图
    cv2.imshow('识别结果', result_image)
    
    print("\n按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果图片
    output_path = image_path.replace('.', '_result.')
    cv2.imwrite(output_path, result_image)
    print(f"\n结果图片已保存至: {output_path}")

def batch_ocr(image_folder):
    """
    批量处理文件夹中的图片
    
    Args:
        image_folder: 图片文件夹路径
    """
    import os
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    # 初始化 EasyOCR
    reader = easyocr.Reader(['ch_sim', 'en'])
    
    # 遍历文件夹
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(supported_formats):
            image_path = os.path.join(image_folder, filename)
            print(f"\n处理图片: {filename}")
            
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图片: {filename}")
                continue
            
            # 进行OCR识别
            results = reader.readtext(image)
            
            # 输出结果
            print(f"在 {filename} 中识别到 {len(results)} 个文本:")
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"  {i+1}. {text} (置信度: {confidence:.2f})")

def save_results_to_file(results, output_file="ocr_results.txt"):
    """
    将识别结果保存到文件
    
    Args:
        results: OCR识别结果
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OCR识别结果\n")
        f.write("=" * 50 + "\n")
        for i, (bbox, text, confidence) in enumerate(results):
            f.write(f"文本 {i+1}: {text}\n")
            f.write(f"置信度: {confidence:.2f}\n")
            f.write(f"位置: {bbox}\n")
            f.write("-" * 30 + "\n")
    
    print(f"识别结果已保存到: {output_file}")

if __name__ == "__main__":
    # 使用示例
    
    # 1. 单张图片识别
    print("=== EasyOCR 文本识别演示 ===")
    
    # 请修改为您的图片路径
    image_path = "debug_processed_3_quantity.png"  # 替换为实际图片路径
    
    try:
        ocr_demo(image_path)
    except FileNotFoundError:
        print(f"图片文件未找到: {image_path}")
        print("请将图片放在项目根目录下，或修改 image_path 变量")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
    
    # 2. 批量处理示例（可选）
    # batch_ocr("images_folder")