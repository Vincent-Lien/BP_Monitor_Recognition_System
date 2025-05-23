import cv2
from ultralytics import YOLO
import os
import shutil

def localization_and_crop_image(
    img_path, 
    model_path='checkpoints/localization_best.pt',
    save_dir='cropped_objects',
    class_names=None,
    save_classes=None,
    conf_start=0.50,
    conf_min=0.05,
    imgsz=640,
    device='0'
):
    # Set default values if not provided
    if class_names is None:
        class_names = ['DIA', 'PUL', 'SYS', 'Sphygmomanometer']
    
    if save_classes is None:
        save_classes = [0, 1, 2]  # 0: DIA, 1: PUL, 2: SYS
    
    # Load model
    model = YOLO(model_path)
    
    # Read image
    img = cv2.imread(img_path)
    
    conf = conf_start
    successful = False
    
    while conf >= conf_min and not successful:
        print(f"Trying with confidence threshold: {conf:.2f}")
        
        # Clear previous results
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        # Perform inference
        results = model(img_path, imgsz=imgsz, conf=conf, device=device)
        
        # Create directories for each class
        for cls in [class_names[i] for i in save_classes]:
            class_dir = os.path.join(save_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
        
        # Process results
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for j, (box, cls) in enumerate(zip(boxes, classes)):
                if cls in save_classes:
                    x1, y1, x2, y2 = map(int, box)
                    crop_img = img[y1:y2, x1:x2]
                    
                    class_name = class_names[cls]
                    class_dir = os.path.join(save_dir, class_name)
                    
                    crop_path = os.path.join(class_dir, f'crop_{i}_{j}.jpg')
                    cv2.imwrite(crop_path, crop_img)
        
        # Check if each folder has exactly one image
        all_folders_have_one_image = True
        for cls in [class_names[i] for i in save_classes]:
            class_dir = os.path.join(save_dir, cls)
            num_images = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if num_images != 1:
                all_folders_have_one_image = False
                print(f"Folder {cls} has {num_images} images, expecting 1")
                break
        
        if all_folders_have_one_image:
            successful = True
            print(f"Successfully found one image for each class with conf={conf:.2f}")
        else:
            conf -= 0.01
    
    if not successful:
        print(f"Could not find exactly one image for each class. Minimum conf {conf_min} reached.")
    
    return conf

# Example usage
if __name__ == "__main__":
    img_path = 'test_image.png'
    final_conf = localization_and_crop_image(img_path)
    print(f"Final confidence threshold: {final_conf:.2f}")