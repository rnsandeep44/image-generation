import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load SAM2 model
checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Function to handle clicks and collect points and labels
# def on_click(event, points, labels):
#     if event.inaxes:
#         points.append([event.xdata, event.ydata])
#         labels.append(1)  # Assuming label 1 for positive clicks


def on_click(event, points, labels):
    if event.inaxes is not None:
        points.append([event.xdata, event.ydata])
        labels.append(1)
        plt.close()

# Function to save mask to disk
def save_mask(mask, save_path):
    mask_image = Image.fromarray((mask * 255).astype('uint8'))
    mask_image.save(save_path)

# Main function to process the image and save masks
def generate_and_save_mask(image_path, save_path):
    image = Image.open(image_path).convert("RGB")

    points, labels = [], []

    # First Point
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, labels))
    plt.title("Click on the image to add the first point")
    plt.show()

    # Second Point
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, points, labels))
    plt.title("Click on the image to add the second point")
    plt.show()

    # Set image for the predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
        )

    # Assuming the first mask is the one we want to save
    mask = masks[0]#.cpu().numpy()

    # Save mask to disk
    save_mask(mask, save_path)

    # Display the mask on the original image
    plt.figure(figsize=(12, 8))
    plt.title("Generated Mask")
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)  # Overlay the mask with transparency
    plt.show()


import sys
# Example usage
image_path = sys.argv[1] #"path/to/image.png"
save_path = sys.argv[2] #"path/to/save/mask.png"

generate_and_save_mask(image_path, save_path)

