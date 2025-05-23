import json
from google import genai

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)
    API_KEY = config['gemini_api_key']
    client = genai.Client(api_key=API_KEY)

def get_health_advice(message, bp_context):
    """
    Use Gemini API to provide health advice based on user message and blood pressure context.
    :param message: User input message
    :param bp_context: Blood pressure data context
    :return: Health advice or response
    """

    prompt = f"""
    You are a health assistant specializing in blood pressure management, providing accurate and concise advice. Your responses should be brief and conversational, using Traditional Chinese.

    Based on the user's latest blood pressure data: {bp_context}, provide tailored advice. If the query is not health-related, respond appropriately in a conversational manner.

    Example:
    User: 我的血壓140/90，該怎麼辦？
    Assistant: 您的血壓有點高喔！建議您諮詢醫生，並考慮調整飲食和增加運動。

    User: 今天天氣真好！
    Assistant: 是啊，今天天氣真不錯呢！

    Query: {message}
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
    )
    
    return response.text.split('Assistant: ')[-1].strip()