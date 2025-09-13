import cv2
import numpy as np
import matplotlib.pyplot as plt
#from collections import Counter
from ultralytics.utils.plotting import Annotator, colors

def create_visualization(original_img, mask_img, warped_img, detections_warped, detections_original, cls_names=None):
    """
    Create a visualization with 5 sections:
    1. Title
    2-4. Three images side by side (segmentation, warped detections, original detections)
    5. Number of detections on original image
    
    Parameters:
    - original_img: OpenCV Mat of the original image
    - mask_img: OpenCV Mat of the segmentation mask (binary)
    - warped_img: OpenCV Mat of the warped image slice
    - detections_warped: List of detections on warped image in YOLOv5 format [x, y, w, h, conf, class]
    - detections_original: List of detections on original image in YOLOv5 format [x, y, w, h, conf, class]
    
    Returns:
    - visualization: OpenCV Mat of the complete visualization
    """
    # Resize mask to match original image if needed
    if mask_img.shape[:2] != original_img.shape[:2]:
        mask_img = cv2.resize(mask_img, (original_img.shape[1], original_img.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
   
    #NOTE the mask_img consists of zeros and 1.0s.
    _, binary_mask = cv2.threshold(mask_img, 0.5, 1, cv2.THRESH_BINARY)
    
    # Create segmentation visualization
    segmentation_vis = original_img.copy()
    # Create colored overlay for the mask (green for planks)
    overlay = np.zeros_like(segmentation_vis)
    overlay[binary_mask > 0] = [0, 255, 0]  # Green color for planks
    # Blend the overlay with the original image
    segmentation_vis = cv2.addWeighted(segmentation_vis, 0.7, overlay, 0.3, 0)
    
    # Draw detections on warped image
    warped_vis = warped_img.copy()
    annotator = Annotator(warped_vis, line_width=2, example=cls_names)
    for (x_min, y_min, x_max, y_max), conf, cls in detections_warped:
        # ensure ints for drawing
        xy = [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))]
        label = f"{cls_names[cls]} {conf:.2f}" if cls_names else f"{cls} {conf:.2f}"
        annotator.box_label(xy, label, color=colors(cls, True))

    # Draw detections on original image
    original_vis = original_img.copy()
    annotator = Annotator(original_vis, line_width=2, example=cls_names)
    for (x_min, y_min, x_max, y_max), conf, cls in detections_original:
        # ensure ints for drawing
        xy = [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))]
        label = f"{cls_names[cls]} {conf:.2f}" if cls_names else f"{cls} {conf:.2f}"
        annotator.box_label(xy, label, color=colors(cls, True))

    # Resize all images to the same size for display
    max_height = 480
    max_width = 480
    
    # Resize images to have the same dimensions
    segmentation_vis_resized = cv2.resize(segmentation_vis, (max_width, max_height))
    warped_vis_resized = cv2.resize(warped_vis, (max_width, max_height))
    original_vis_resized = cv2.resize(original_vis, (max_width, max_height))
    
    # Create title section
    title_height = 60
    title_section = np.ones((title_height, max_width * 3, 3), dtype=np.uint8) * 255
    cv2.putText(title_section, "Plank Anomaly Detection", 
                (max_width * 3 // 2 - 200, title_height // 2 + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Create bottom section with detection count
    bottom_height = 60
    bottom_section = np.ones((bottom_height, max_width * 3, 3), dtype=np.uint8) * 255
    detection_text = f"Number of detected anomalies: {len(detections_original)}"
    cv2.putText(bottom_section, detection_text, 
                (max_width * 3 // 2 - 200, bottom_height // 2 + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add labels to each image
    cv2.putText(segmentation_vis_resized, "Segmentation Result", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(warped_vis_resized, "Warped Image Detections", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(original_vis_resized, "Original Image Detections", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Concatenate the three images horizontally
    middle_section = np.hstack((segmentation_vis_resized, warped_vis_resized, original_vis_resized))
    
    # Concatenate all sections vertically
    visualization = np.vstack((title_section, middle_section, bottom_section))
    
    return visualization

def show_visualization(visualization):
    """Display the visualization using OpenCV"""
    cv2.imshow("Plank Detection Visualization", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_visualization(visualization, output_path="visualization_result.jpg"):
    """Save the visualization to a file"""
    cv2.imwrite(output_path, visualization)
    print(f"Visualization saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    # These would be replaced with your actual inputs
    original_img = cv2.imread("original.jpg")
    mask_img = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
    warped_img = cv2.imread("warped.jpg")
    
    # Example detections in YOLOv5 format [x, y, w, h, conf, class]
    detections_warped = [
        [0.5, 0.5, 0.2, 0.3, 0.95, 0],
        [0.7, 0.6, 0.15, 0.25, 0.87, 0]
    ]
    
    detections_original = [
        [0.3, 0.4, 0.1, 0.2, 0.92, 0],
        [0.5, 0.6, 0.12, 0.18, 0.85, 0],
        [0.7, 0.5, 0.14, 0.22, 0.78, 0]
    ]
    
    # Create and display the visualization
    vis = create_visualization(original_img, mask_img, warped_img, 
                              detections_warped, detections_original)
    show_visualization(vis)
    save_visualization(vis)
