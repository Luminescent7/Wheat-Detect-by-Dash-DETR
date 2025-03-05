from ultralytics import YOLO
import numpy as np

# 加载 YOLO 模型（请确保提供正确的模型权重路径）
model = YOLO('best.pt')  # 假设模型权重文件名为 best.pt，如有需要可修改路径

def detect(image, conf=0.5, iou=0.5):
    """
    使用 YOLO 模型对给定图像进行检测。
    参数:
        image: 图像路径或图像数据（PIL 图像或 numpy 数组）。
        conf: 置信度阈值。
        iou: IoU 阈值。
    返回:
        annotated_img (np.ndarray): 带检测框和标签的图像（numpy 数组，BGR格式）。
        detections (list): 检测到的目标列表，每个元素包含 'class' 和 'confidence'。
    """
    try:
        # 执行模型推理
        results = model.predict(image, conf=conf, iou=iou)
        if len(results) == 0:
            return None, []  # 没有结果
        result = results[0]
        # 获取带标注的图像（绘制了边界框和类别）
        annotated_img = result.plot()  # 返回带检测标注的图像数组 (BGR 格式)
        # 提取检测信息（类别和置信度）
        detections = []
        for box in result.boxes:
            conf_val = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
            class_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
            class_name = result.names.get(class_id, class_id) if hasattr(result, 'names') else class_id
            detections.append({
                'class': class_name,
                'confidence': conf_val
            })
        return annotated_img, detections
    except Exception as e:
        return None, [{'error': str(e)}]
