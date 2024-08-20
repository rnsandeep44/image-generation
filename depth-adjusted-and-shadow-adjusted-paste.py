import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load MiDaS model
model_type = "DPT_Large"  # Other options: "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()


def horizontal_concatenate(image1, image2, output_path):
    # Load the two images

    # Ensure the images have the same height by resizing if necessary
    if image1.shape[0] != image2.shape[0]:
        height = min(image1.shape[0], image2.shape[0])
        image1 = cv2.resize(image1, (int(image1.shape[1] * height / image1.shape[0]), height))
        image2 = cv2.resize(image2, (int(image2.shape[1] * height / image2.shape[0]), height))

    # Concatenate the images horizontally
    result = cv2.hconcat([image1, image2])

    # Save the result
    cv2.imwrite(output_path, result)
    return

# Function to estimate depth
def estimate_depth(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (1280, 720))
    print("image:", image_path, "size:", img.shape)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms.default_transform(rgb_img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map, normalized_depth_map, img

def get_light_direction(image):
    # Load the image
      # 'your_scene_image.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection to find shadows
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Use Hough Transform to find lines in the image
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Draw lines on the image and estimate light direction
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Average direction of detected lines
        avg_theta = np.mean(lines[:, 0, 1])
        light_direction = (np.cos(avg_theta), np.sin(avg_theta))

        print(f"Estimated light direction: {light_direction}")
        return light_direction



def add_upside_down_shadow(scene_img, person_mask, position_x, position_y, shadow_opacity=0.5, blur_kernel=(21, 21), sigma=10, shadow_length_factor=1.0):


    # Calculate the bounding box of the masked region
    ys, xs = np.where(person_mask > 0)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    # Extract the region of interest (ROI) from the person mask
    roi_person_mask = person_mask[min_y:max_y+1, min_x:max_x+1]

    new_size = (int(roi_person_mask.shape[1] * shadow_length_factor), int(roi_person_mask.shape[0] * shadow_length_factor))
    roi_person_mask = cv2.resize(roi_person_mask, new_size, interpolation=cv2.INTER_AREA)

    # Flip the ROI of the person mask vertically to create the upside-down shadow
    flipped_person_mask = cv2.flip(roi_person_mask, 0)

    # Create a shadow mask with the same size as the scene image
    shadow_mask = np.zeros_like(scene_img)

    # Calculate the region where the shadow will be placed
    mask_h, mask_w = flipped_person_mask.shape[:2]

    print("flipped mask shape:", mask_h, mask_w)
    start_x = max(position_x , 0)
    start_y = max(position_y, 0)
    end_x = min(start_x + mask_w, scene_img.shape[1])
    end_y = min(start_y + mask_h, scene_img.shape[0])

    # Calculate valid width and height within scene bounds
    valid_width = end_x - start_x
    valid_height = end_y - start_y

    # Debug: Print the shadow position and size
    print(f"Shadow Position: start_x={start_x}, start_y={start_y}, end_x={end_x}, end_y={end_y}")
    print(f"Flipped Person Mask Size: width={mask_w}, height={mask_h}")
    print(f"Valid Region Size: width={valid_width}, height={valid_height}")
    print(f"Scene Image Size: width={scene_img.shape[1]}, height={scene_img.shape[0]}")

    # Check if the shadow region is valid
    if valid_width > 0 and valid_height > 0:
        shadow_region = shadow_mask[start_y:end_y, start_x:end_x]

        # Apply the flipped ROI person mask to the shadow region
        for c in range(0, 3):  # Apply to each color channel
            shadow_region[:, :, c] = flipped_person_mask[:valid_height, :valid_width] * 255

        # Debug: Check the values in the shadow region before blurring
        print(f"Shadow Region Non-Zero Values Before Blurring: {np.count_nonzero(shadow_region)}")

        # Blur the shadow to soften it
        shadow_mask_blurred = cv2.GaussianBlur(shadow_mask, blur_kernel, sigmaX=sigma, sigmaY=sigma)

        # Debug: Check if the shadow mask is being applied correctly after blurring
        print(f"Shadow Mask Applied After Blurring: {np.any(shadow_mask_blurred)}")

        # Apply the shadow on the scene with appropriate opacity
        scene_img_with_shadow = cv2.addWeighted(scene_img, 1.0, shadow_mask_blurred, -shadow_opacity, 0)
    else:
        # If the shadow region is out of bounds, return the original image
        scene_img_with_shadow = scene_img
        print("Shadow region is out of bounds or invalid. No shadow applied.")

    return scene_img_with_shadow




# Click event handler for matplotlib
def on_click(event):
    global position_x, position_y
    if event.inaxes:
        position_x, position_y = int(event.xdata), int(event.ydata)
        plt.close()  # Close the figure after a click

# Load scene and person images
scene_img_path = sys.argv[1]  # Path to the scene image
person_img_path = sys.argv[2] #'runner.jpg'  # Path to the person image
person_mask_path = sys.argv[3] #'person_mask.jpg'  # Path to the person mask


original_scene_image = cv2.imread(scene_img_path)


# Estimate depth for both scene and person
scene_depth_map, scene_depth_map_normalized, scene_img = estimate_depth(scene_img_path)
person_depth_map, person_depth_map_normalized, person_img = estimate_depth(person_img_path)

# Load the person mask
person_mask = cv2.imread(person_mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
person_mask = cv2.resize(person_mask, (1280, 720))

# Calculate the bounding box of the masked region
ys, xs = np.where(person_mask > 0)
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

# Extract the region of interest (ROI) from the person image and mask
roi_person_img = person_img[min_y:max_y+1, min_x:max_x+1]
roi_person_mask = person_mask[min_y:max_y+1, min_x:max_x+1]

# Update person width and height based on the ROI
person_height, person_width = roi_person_mask.shape

# Display the scene image and wait for user to click
fig, ax = plt.subplots()
ax.imshow(cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB))
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.title('Click on the scene to position the person')
plt.show()

# Calculate the placement region on the scene depth map based on the person size
placement_region = scene_depth_map[position_y:position_y + person_height, position_x:position_x + person_width]

# Calculate the median depth of the placement region in the scene
placement_median_depth = np.median(placement_region)

# Calculate the median depth of the person (only in the masked region)
person_depth_values = person_depth_map[min_y:max_y+1, min_x:max_x+1][roi_person_mask > 0]
person_median_depth = np.median(person_depth_values)

# Calculate scale factor based on the relative depth
scale_factor = placement_median_depth / person_median_depth

# Resize the person image and mask based on the calculated scale factor
new_size = (int(roi_person_img.shape[1] * scale_factor), int(roi_person_img.shape[0] * scale_factor))
resized_person = cv2.resize(roi_person_img, new_size, interpolation=cv2.INTER_AREA)
resized_mask = cv2.resize(roi_person_mask, new_size, interpolation=cv2.INTER_AREA)

# Adjust the placement position based on the resized dimensions
position_x = max(0, min(scene_img.shape[1] - resized_person.shape[1], position_x))
position_y = max(0, min(scene_img.shape[0] - resized_person.shape[0], position_y))

#bottom_middle_x = position_x + resized_person.shape[1] // 2
#bottom_middle_y = position_y + resized_person.shape[0]

#bottom_middle_point = (bottom_middle_x, bottom_middle_y)

# Place the person in the scene using the mask for blending
for c in range(0, 3):
    scene_img[position_y:position_y + resized_person.shape[0], position_x:position_x + resized_person.shape[1], c] = (
        resized_mask * resized_person[:, :, c] + 
        (1 - resized_mask) * scene_img[position_y:position_y + resized_person.shape[0], position_x:position_x + resized_person.shape[1], c]
    )

top_left = (position_x, position_y)
bottom_right = (position_x + resized_person.shape[1], position_y + resized_person.shape[0])


print("top left:", top_left)
print("bootm right:", bottom_right)
bottom_left_point = ( top_left[0], bottom_right[1] )

#cv2.rectangle(scene_img, top_left, bottom_right, (0, 255, 0), 2)


position_x = bottom_left_point[0]  # Example position
position_y = bottom_left_point[1]  # Example position
print("position:", position_x, position_y)
# light_direction = get_light_direction(image=scene_img.copy())
# #print(light_direction)


mask = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE) #/ 255.0
mask = cv2.resize(mask, (1280, 720)) / 255.0

shadow_needed = True
if shadow_needed:
    scene_img_with_shadow = add_upside_down_shadow(scene_img, mask, position_x, position_y,
                                                   shadow_length_factor=scale_factor)
    # Save or display the resulting image
    # output_img = cv2.cvtColor(scene_img_with_shadow, cv2.COLOR_BGR2RGB)
    output_img = scene_img_with_shadow
    cv2.imwrite('output_with_person_shadow.jpg', scene_img_with_shadow)

else:
    output_img = scene_img
    # output_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('output_with_person.jpg', scene_img)



horizontal_concatenate(original_scene_image, output_img, "result_"+ scene_img_path)




# Display the result
plt.figure(figsize=(10, 5))
matplotlib_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
plt.imshow(matplotlib_img)
plt.title('Person Placed in Scene')
plt.axis('off')
plt.show()

