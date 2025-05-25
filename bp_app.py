import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

from utils.image_preprocess import adjust_image
from utils.inference_localization import localization_and_crop_image
from utils.inference_seven_seg_classification import inference_single_image
from utils.cut_digits import cut_out_2_numbers, cut_out_3_numbers

# 全局列表，用於存儲血壓記錄
bp_records = [] # 每個元素: {'timestamp': datetime, 'SYS': int, 'DIA': int, 'PUL': int}

def process_bp_image_and_update(image_filepath):
    """
    處理上傳的血壓計圖片（預期為文件路徑），提取讀數，存儲它們，並生成圖表更新數據。
    """
    global bp_records
    if image_filepath is None:
        return gr.update(value=None), create_bp_plot(), "請上傳圖片以開始。"

    try:
        # 您的代碼: localization_and_crop_image(image) 其中 image 是文件路徑
        # Gradio gr.Image(type="filepath") 提供文件路徑
        result_images = localization_and_crop_image(image_filepath)

        sys_image = result_images.get('SYS')
        dia_image = result_images.get('DIA')
        pul_image = result_images.get('PUL')

        if sys_image is None or dia_image is None or pul_image is None:
            raise ValueError("無法從圖片中定位所有必要的數值區域 (SYS, DIA, PUL)。請確認圖片清晰且完整。")

        if not all(isinstance(img, np.ndarray) and img.size > 0 for img in [sys_image, dia_image, pul_image]):
            raise ValueError("圖像裁切後格式不正確或為空。")
        
        # SYS 圖片去除上方陰影
        if sys_image.shape[0] == 0 or sys_image.shape[1] == 0:
             raise ValueError("SYS 區域裁切結果無效 (維度為0)。")
        crop_height = int(sys_image.shape[0] * 0.07)
        # 確保裁切不會導致圖像為空
        sys_image_processed = sys_image[crop_height:, :, :] if sys_image.shape[0] > crop_height else sys_image

        if sys_image_processed.size == 0: sys_image_processed = sys_image # 如果裁切後為空，則回退

        # DIA, PUL 調整圖片
        dia_image_processed = adjust_image(dia_image, brightness_factor=1.5, contrast_factor=2.0, sharpness_factor=2.0, color_factor=1.0)
        pul_image_processed = adjust_image(pul_image, brightness_factor=2.0, contrast_factor=2.0, sharpness_factor=2.0, color_factor=1.0)

        sys_number_str, dia_number_str, pul_number_str = '', '', ''

        # 判斷 SYS 是否為三位數
        if dia_image_processed.shape[1] == 0: # 檢查分母是否為0
            raise ValueError("DIA 區域裁切後寬度為0，無法判斷 SYS 位數。")
        
        # 確保 sys_image_processed 也有有效寬度
        if sys_image_processed.shape[1] == 0:
            raise ValueError("SYS 區域處理後寬度為0。")

        if sys_image_processed.shape[1] / dia_image_processed.shape[1] > 1.05:
            sys_digits = cut_out_3_numbers(sys_image_processed)
        else:
            sys_digits = cut_out_2_numbers(sys_image_processed)

        for digit in sys_digits:
            if not isinstance(digit, np.ndarray) or digit.size == 0: continue
            prediction, _ = inference_single_image(digit)
            sys_number_str += str(prediction)
        sys_number = int(sys_number_str) if sys_number_str else 0

        # DIA 數字辨識
        dia_digits = cut_out_2_numbers(dia_image_processed)
        for digit in dia_digits:
            if not isinstance(digit, np.ndarray) or digit.size == 0: continue
            prediction, _ = inference_single_image(digit)
            dia_number_str += str(prediction)
        dia_number = int(dia_number_str) if dia_number_str else 0

        # PUL 數字辨識
        if pul_image_processed.size > 0 and pul_image_processed.shape[0] > 0 and pul_image_processed.shape[1] > 0:
            pul_digits = cut_out_2_numbers(pul_image_processed)
            for digit in pul_digits:
                if not isinstance(digit, np.ndarray) or digit.size == 0: continue
                prediction, _ = inference_single_image(digit)
                pul_number_str += str(prediction)
            pul_number = int(pul_number_str) if pul_number_str else 0
        else:
            pul_number = 0 # 如果PUL圖像無效，則默認為0

        timestamp = datetime.now()
        bp_records.append({
            'timestamp': timestamp,
            'SYS': sys_number,
            'DIA': dia_number,
            'PUL': pul_number
        })

        plot = create_bp_plot()
        confirmation_message = f"記錄成功 ({timestamp.strftime('%Y-%m-%d %H:%M:%S')}): 收縮壓 {sys_number}, 舒張壓 {dia_number}, 心率 {pul_number}"
        return gr.update(value=None), plot, confirmation_message

    except Exception as e:
        error_message = f"處理圖片時發生錯誤: {str(e)}"
        # 發生錯誤時，清空圖片上傳器，顯示當前圖表（或空圖表），並顯示錯誤消息
        return gr.update(value=None), create_bp_plot(), error_message
    finally:
        # 如果 image_filepath 是臨時文件路徑，Gradio 會處理它的清理
        # 如果您手動保存了臨時文件，請在此處刪除 os.remove(image_filepath)
        pass


def create_bp_plot():
    """
    創建記錄血壓數據的圖表。返回一個 Matplotlib Figure。
    """
    global bp_records
    plt.close('all') # 關閉之前可能存在的圖，避免內存洩漏或重疊

    if not bp_records:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No blood pressure record yet.", ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        fig.patch.set_facecolor('#f0f0f0') # 設置圖表背景色
        ax.set_facecolor('#f0f0f0')
        plt.tight_layout()
        return fig

    # 按時間戳排序記錄
    sorted_records = sorted(bp_records, key=lambda x: x['timestamp'])
    df = pd.DataFrame(sorted_records)

    fig, ax = plt.subplots(figsize=(8, 4)) # 調整圖表大小
    ax.plot(df['timestamp'], df['SYS'], marker='o', linestyle='-', label='Systolic (mmHg)', color='#ff6347', linewidth=2) # 番茄紅
    ax.plot(df['timestamp'], df['DIA'], marker='o', linestyle='-', label='Diastolic (mmHg)', color='#4682b4', linewidth=2) # 鋼青藍
    ax.plot(df['timestamp'], df['PUL'], marker='s', linestyle='--', label='Pulse (bpm)', color='#3cb371', linewidth=2) # 中海綠

    ax.set_xlabel("Record Time", fontsize=12)
    ax.set_ylabel("Measured Value", fontsize=12)
    ax.set_title("Blood Pressure and Pulse Trend Chart", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')

    # 優化X軸日期格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    fig.autofmt_xdate(rotation=30, ha='right') # 自動旋轉日期標籤以獲得更好的適應性

    # 美化外觀
    fig.patch.set_facecolor('#f8f9fa') # 輕微的背景色
    ax.set_facecolor('#ffffff') # 圖表區域背景色
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')


    plt.tight_layout() # 自動調整子圖參數以給出緊湊的佈局
    return fig

# Gradio 界面定義
with gr.Blocks(title="血壓記錄小幫手", theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.cyan)) as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: #ffffff; font-family: 'Arial', sans-serif;">血壓記錄小幫手 🩺</h1>
            <p style="color: #ffffff; font-size: 1.1em;">
                歡迎使用！請上傳您的血壓計螢幕照片，系統將自動辨識讀數並為您繪製健康趨勢圖。
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 步驟一：上傳測量圖片")
            # 使用 type="filepath" 因為您的原始腳本似乎期望一個路徑。
            # Gradio 會處理臨時文件的創建和清理。
            image_uploader = gr.Image(
                type="filepath",
                label="點擊此處上傳或拖曳圖片",
                sources=["upload"], # 僅允許從本地上傳
                height=200, # 限制上傳組件高度
            )
            status_message = gr.Textbox(
                label="系統訊息",
                placeholder="上傳圖片後，此處將顯示處理結果...",
                lines=3, # 允許多行訊息
                interactive=False,
                show_copy_button=True
            )

        with gr.Column(scale=2, min_width=500):
            gr.Markdown("### 步驟二：查看健康趨勢")
            plot_display = gr.Plot(label="血壓歷史紀錄圖")

    # 事件處理：當圖片上傳時，調用處理函數
    # outputs:
    # 1. image_uploader: 清空 (value=None)
    # 2. plot_display: 更新圖表
    # 3. status_message: 顯示確認或錯誤訊息
    image_uploader.upload(
        fn=process_bp_image_and_update,
        inputs=[image_uploader],
        outputs=[image_uploader, plot_display, status_message],
        show_progress="full" # 顯示上傳和處理進度條
    )

    # 應用加載時初始化圖表（顯示 "尚無紀錄" 或任何已有的數據）
    demo.load(fn=create_bp_plot, inputs=None, outputs=plot_display)

if __name__ == '__main__':
    # 要運行此應用：
    # 1. 將此代碼保存為 .py 文件（例如 bp_app.py）。
    # 2. 確保已安裝必要的庫: pip install gradio pandas matplotlib numpy
    # 3. 如果您使用真實的 utils 函數，請確保 'utils' 文件夾和相關腳本
    #    與 bp_app.py 在同一目錄中，或 Python 環境可以找到它們。
    #    並移除或註釋掉此文件中的模擬函數定義和賦值。
    # 4. 從終端運行: python bp_app.py
    demo.launch()
    # demo.launch(share=True) # 如果需要公開鏈接，請使用 share=True