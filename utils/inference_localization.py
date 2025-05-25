import cv2
from ultralytics import YOLO

def localization_and_crop_image(
    img_path, 
    model_path='checkpoints/localization_best.pt',
    class_names=None,
    save_classes=None,
    conf_start=0.46,
    conf_min=0.05,
    imgsz=640,
    device='0',
    verbose=False
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
    cropped_images = {}  # Dictionary to store cropped images
    
    while conf >= conf_min and not successful:
        if verbose:
            print(f"Trying with confidence threshold: {conf:.2f}")
        
        # Clear previous results
        cropped_images = {}
        
        # Perform inference
        results = model(img_path, imgsz=imgsz, conf=conf, device=device, verbose=verbose)
        
        # Process results
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            
            for j, (box, cls) in enumerate(zip(boxes, classes)):
                if cls in save_classes:
                    x1, y1, x2, y2 = map(int, box)
                    crop_img = img[y1:y2, x1:x2]
                    
                    class_name = class_names[cls]
                    if class_name not in cropped_images:
                        cropped_images[class_name] = []
                    
                    cropped_images[class_name].append(crop_img)
        
        # Check if each class has exactly one image
        all_classes_have_one_image = True
        for cls in [class_names[i] for i in save_classes]:
            if cls not in cropped_images or len(cropped_images[cls]) != 1:
                images_count = 0 if cls not in cropped_images else len(cropped_images[cls])
                all_classes_have_one_image = False
                if verbose:
                    print(f"Class {cls} has {images_count} images, expecting 1")
                break
        
        if all_classes_have_one_image:
            successful = True
            if verbose:
                print(f"Successfully found one image for each class with conf={conf:.2f}")
        else:
            conf -= 0.01
    
    if not successful:
        if verbose:
            print(f"Could not find exactly one image for each class. Minimum conf {conf_min} reached.")
        return None
    
    # Extract the single image for each class
    # Extract the single image for each class in the specified order
    result_images = {}
    for cls in ['SYS', 'DIA', 'PUL']:
        if cls in cropped_images and len(cropped_images[cls]) > 0:
            result_images[cls] = cropped_images[cls][0]
    
    return result_images