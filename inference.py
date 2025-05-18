import cv2
from ultralytics import YOLO
import os

# 类别名称列表
CLASS_NAMES = ['DLA', 'PUL', 'SYS', 'Sphygmomanometer']
# 希望保存的类别索引 (前三个)
SAVE_CLASSES = [0, 1, 2]  # 0: DLA, 1: PUL, 2: SYS

# 加载训练好的模型权重
model = YOLO('runs/detect/train/weights/best.pt')

# 读取原始图像
img_path = 'dataset/test/images/91_jpeg.rf.22b311be3d7381b1c5477da0b4358ef7.jpg'
img = cv2.imread(img_path)

# 进行推理
results = model(img_path, imgsz=640, conf=0.25, device='0')

# 创建保存裁剪图的目录
save_dir = 'cropped_objects'
os.makedirs(save_dir, exist_ok=True)

for i, result in enumerate(results):
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    for j, (box, cls) in enumerate(zip(boxes, classes)):
        # 检查类别是否需要保存
        if cls in SAVE_CLASSES:
            x1, y1, x2, y2 = map(int, box)
            crop_img = img[y1:y2, x1:x2]

            # 使用类别名称作为文件夹名
            class_name = CLASS_NAMES[cls]
            class_dir = os.path.join(save_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            crop_path = os.path.join(class_dir, f'crop_{i}_{j}.jpg')
            cv2.imwrite(crop_path, crop_img)

print(f'指定类别的裁剪图已分别保存到 {save_dir} 下的子文件夹中')