import os 

from utils.image_preprocess import adjust_image
from utils.inference_localization import localization_and_crop_image
from utils.inference_seven_seg_classification import inference_single_image
from utils.cut_digits import cut_out_2_numbers, cut_out_3_numbers

image_folder = 'my_images'
for image in sorted(os.listdir(image_folder)):

    image = os.path.join(image_folder, image)

    # 切出SYS、DIA、PUL三個區域
    result_images = localization_and_crop_image(image)
    
    sys_image = result_images.get('SYS')
    dia_image = result_images.get('DIA')
    pul_image = result_images.get('PUL')

    # SYS 圖片去除上方陰影
    crop_height = int(sys_image.shape[0] * 0.07)
    sys_image = sys_image[crop_height:, :, :]

    # DIA, PUL 調整圖片亮度、對比度、銳利度和色彩飽和度
    dia_image = adjust_image(dia_image, brightness_factor=1.5, contrast_factor=2.0, sharpness_factor=2.0, color_factor=1.0)
    pul_image = adjust_image(pul_image, brightness_factor=2.0, contrast_factor=2.0, sharpness_factor=2.0, color_factor=1.0)
    
    # 辨識圖片數字
    sys_number, dia_number, pul_number = '', '', ''

    # 判斷 SYS 是否為三位數、分離各位數字
    if sys_image.shape[1] / dia_image.shape[1] > 1.05:
        sys_digits = cut_out_3_numbers(sys_image)
    else:
        sys_digits = cut_out_2_numbers(sys_image)

    # SYS 數字辨識
    for digit in sys_digits:
        prediction, _ = inference_single_image(digit)
        sys_number += str(prediction)
    sys_number = int(sys_number)
    
    # DIA 數字辨識
    dia_digits = cut_out_2_numbers(dia_image)
    for digit in dia_digits:
        prediction, _ = inference_single_image(digit)
        dia_number += str(prediction)
    dia_number = int(dia_number)

    # PUL 數字辨識
    pul_digits = cut_out_2_numbers(pul_image)
    for digit in pul_digits:
        prediction, _ = inference_single_image(digit)
        pul_number += str(prediction)
    pul_number = int(pul_number)
    
    print(f"image: {image}, SYS: {sys_number}, DIA: {dia_number}, PUL: {pul_number}")