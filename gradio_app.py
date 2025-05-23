import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import glob
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import shutil

from utils.inference_localization import localization_and_crop_image
from utils.number_recognition import get_number_from_image
from utils.chatbot import get_health_advice

# Function to process uploaded image and save to CSV
def process_image(image):
    if image is None:
        return "No image uploaded.", None
    
    # Save uploaded image temporarily
    image_path = "temp_image.png"
    image.save(image_path)

    # Run localization and cropping
    localization_and_crop_image(image_path)

    cropped_objects_dir = 'cropped_objects'
    subdirs = ['SYS', 'DIA', 'PUL']
    results = {}

    # Process each subdirectory
    for subdir in subdirs:
        subdir_path = os.path.join(cropped_objects_dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Directory {subdir_path} does not exist, skipping.")
            continue
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(os.path.join(subdir_path, ext)))
        
        if not image_files:
            print(f"No images found in {subdir_path}, skipping.")
            continue
        
        image_files.sort()
        number = get_number_from_image(image_files[0])
        results[subdir] = number

    # Add timestamp
    results['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV
    csv_file = 'bp_readings.csv'
    file_exists = os.path.isfile(csv_file)
    fieldnames = ['Timestamp', 'SYS', 'DIA', 'PUL']
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    # Clean up
    if os.path.exists(cropped_objects_dir):
        shutil.rmtree(cropped_objects_dir)
    
    # Generate chart after processing
    chart_path = generate_weekly_chart()
    
    return f"Image processed. Results saved to database.", chart_path

# Function to generate weekly blood pressure chart
def generate_weekly_chart():
    csv_file = 'bp_readings.csv'
    if not os.path.isfile(csv_file):
        return None  # Return None if no data exists

    # Read CSV
    df = pd.read_csv(csv_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Get current week's Sunday to Saturday
    today = datetime.now()
    start_of_week = today - timedelta(days=today.weekday() + 1)  # Sunday
    end_of_week = start_of_week + timedelta(days=6)  # Saturday
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_week = end_of_week.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    # Filter data for current week
    mask = (df['Timestamp'] >= start_of_week) & (df['Timestamp'] <= end_of_week)
    weekly_data = df[mask]
    
    if weekly_data.empty:
        return None  # Return None if no data for the week
    
    # Create line chart
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data['Timestamp'], weekly_data['SYS'], label='SYS', marker='o')
    plt.plot(weekly_data['Timestamp'], weekly_data['DIA'], label='DIA', marker='o')
    plt.plot(weekly_data['Timestamp'], weekly_data['PUL'], label='PUL', marker='o')
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.title('Weekly Blood Pressure Readings')
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    chart_path = 'weekly_bp_chart.png'
    plt.savefig(chart_path)
    plt.close()
    
    return chart_path

# Function to interact with chatbot
def chatbot_interaction(message, chat_history):
    if not message:
        return chat_history  # Return unchanged history if no message
    
    # Prepare blood pressure data for context
    csv_file = 'bp_readings.csv'
    bp_context = ""
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        if not df.empty:
            latest = df.iloc[-1]
            bp_context = f"Latest blood pressure: SYS={latest['SYS']}, DIA={latest['DIA']}, PUL={latest['PUL']} at {latest['Timestamp']}"
    
    # Get response from chatbot
    response = get_health_advice(message, bp_context)
    
    # Append to chat history
    chat_history = chat_history or []  # Initialize if None
    chat_history.append([message, response])
    
    return chat_history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Blood Pressure Monitoring App")
    
    with gr.Row():
        # Left column: Image upload and chart
        with gr.Column(scale=1):
            # Top: Image upload section
            gr.Markdown("## Upload Blood Pressure Monitor Image")
            image_input = gr.Image(type="pil", label="Drag and drop an image")
            output_text = gr.Textbox(label="Processing Result")
            
            # Bottom: Weekly chart section
            gr.Markdown("## Weekly Blood Pressure History")
            chart_output = gr.Image(label="Weekly Chart")
        
        # Right column: Chatbot
        with gr.Column(scale=1):
            gr.Markdown("## Health Advice Chatbot")
            chatbot = gr.Chatbot(label="Chat with Health Assistant")
            message_input = gr.Textbox(label="Ask a question or get advice", placeholder="Type your message here...")
            chat_button = gr.Button("Send")
    
    # Event handlers
    image_input.upload(fn=process_image, inputs=image_input, outputs=[output_text, chart_output])
    chat_button.click(fn=chatbot_interaction, inputs=[message_input, chatbot], outputs=chatbot)

demo.launch()