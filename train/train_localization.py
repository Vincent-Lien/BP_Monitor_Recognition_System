from ultralytics import YOLO

# 加载模型，支持预训练权重或自定义模型yaml
model = YOLO('yolo11n.pt')  # 或者YOLO('yolo11.yaml')

# 开始训练
model.train(
    data='dataset/localization/data.yaml',    # 数据集配置文件路径
    epochs=50,           # 训练轮数
    imgsz=640,           # 输入图像大小
    batch=16,             # 批量大小
    device='0',          # 使用GPU编号，CPU用 'cpu'
    workers=4,           # 数据加载线程数
    amp=True             # 是否启用混合精度训练
)
