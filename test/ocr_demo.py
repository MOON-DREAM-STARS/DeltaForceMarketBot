from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = r"C:\Users\admin\Desktop\Github\DeltaForceMarketBot\debug_preprocess_enhanced_1.png"
result = ocr.predict(img_path)

print("OCR识别结果:")
# 从字典中提取识别结果
texts = result[0]['rec_texts']
scores = result[0]['rec_scores']
boxes = result[0]['rec_polys']

for i, (text, score) in enumerate(zip(texts, scores)):
    print(f"text: {text}\tconfidence: {score:.3f}")

# 可视化并保存
image = cv2.imread(img_path)
for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
    pts = np.array(box, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
    cv2.putText(image, text, (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("result.png", image)