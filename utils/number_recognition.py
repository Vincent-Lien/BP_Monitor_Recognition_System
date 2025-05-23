import json
from google import genai

with open('config.json', 'r') as f:
    config = json.load(f)
    API_KEY = config['gemini_api_key']
    client = genai.Client(api_key=API_KEY)

def get_number_from_image(image_path):
    """
    使用Gemini API从图像中提取数字
    :param image_path: 图像文件路径
    :return: 提取的数字
    """
    # 上传图像文件
    my_file = client.files.upload(file=image_path)

    # 设置提示语
    prompt = "請判讀以下七段顯示器照片中顯示的所有數字。如果有多個數字，請從左到右按順序判讀並列出，不要有任何額外的說明或解釋。"

    # 调用Gemini API进行内容生成
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, prompt],
    )

    # 返回响应文本
    return response.text.replace("\n", "")

if __name__ == "__main__":
    # 示例图像路径
    image_path = 'cropped_objects/DLA/crop_0_1.jpg'
    
    # 调用函数并打印结果
    result = get_number_from_image(image_path)
    print(result)