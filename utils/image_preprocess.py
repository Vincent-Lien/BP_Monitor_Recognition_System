from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np

def adjust_image(image, output_path=None, brightness_factor=1.0, contrast_factor=1.0, sharpness_factor=1.0, color_factor=1.0, show=False, save=False):
    """
    調整圖片的亮度、對比度、銳利度和色彩飽和度。

    Args:
        image_path (str): 輸入圖片的路徑。
        output_path (str): 輸出圖片的路徑。
        brightness_factor (float): 亮度調整因子。1.0 表示不變，大於 1.0 變亮，小於 1.0 變暗。
        contrast_factor (float): 對比度調整因子。1.0 表示不變，大於 1.0 增加對比，小於 1.0 降低對比。
        sharpness_factor (float): 銳利度調整因子 (增豔)。1.0 表示不變，大於 1.0 增加銳利度，小於 1.0 模糊。
        color_factor (float): 色彩飽和度調整因子 (增豔)。1.0 表示不變，大於 1.0 增加飽和度，小於 1.0 降低飽和度。
    """
    # 打開圖片
    image = Image.fromarray(image).convert("RGB")  # 確保圖片是 RGB 模式

    # 調整亮度 (亮部)
    enhancer_b = ImageEnhance.Brightness(image)
    img_bright = enhancer_b.enhance(brightness_factor)

    # 調整對比度
    enhancer_c = ImageEnhance.Contrast(img_bright)
    img_contrast = enhancer_c.enhance(contrast_factor)

    # 調整銳利度 (增豔 - 銳利度方面)
    enhancer_s = ImageEnhance.Sharpness(img_contrast)
    img_sharp = enhancer_s.enhance(sharpness_factor)

    # 調整色彩飽和度 (增豔 - 色彩方面)
    enhancer_col = ImageEnhance.Color(img_sharp)
    img_final = enhancer_col.enhance(color_factor)
    
    if show:
        # 展示原圖和調整後的圖片對比
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image))
        plt.title('Original Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(img_final))
        plt.title('Adjusted Image')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    if output_path and save:
        img_final.save(output_path)
    
    return np.array(img_final)  # 返回調整後的圖片數組