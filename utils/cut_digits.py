import cv2
import matplotlib.pyplot as plt 

def cut_out_3_numbers(image, show=False):
    reshaped_image = cv2.resize(image, (1000, 500))
    # Extract three segments of the image based on x-coordinates
    first_segment = reshaped_image[:, 0:210, :]
    second_segment = reshaped_image[:, 210:600, :]
    third_segment = reshaped_image[:, 600:1000, :]

    if show:    
        # Display the segments
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(first_segment, cv2.COLOR_BGR2RGB))
        plt.title('First Segment (0-200)')
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(second_segment, cv2.COLOR_BGR2RGB))
        plt.title('Second Segment (200-600)')
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(third_segment, cv2.COLOR_BGR2RGB))
        plt.title('Third Segment (600-1000)')
        plt.show()

    return first_segment, second_segment, third_segment

def cut_out_2_numbers(image, show=False):
    reshaped_image = cv2.resize(image, (1000, 500))
    # Extract three segments of the image based on x-coordinates
    first_segment = reshaped_image[:, 0:500, :]
    second_segment = reshaped_image[:, 500:1000, :]

    if show:
        # Display the segments
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(first_segment, cv2.COLOR_BGR2RGB))
        plt.title('First Segment (0-500)')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(second_segment, cv2.COLOR_BGR2RGB))
        plt.title('Second Segment (500-1000)')
        plt.show()

    return first_segment, second_segment