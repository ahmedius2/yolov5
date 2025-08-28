import os
import glob
from pathlib import Path

def clean_dataset_remove_unlabeled():
    """
    Remove images that have no corresponding label files or empty label files
    """
    
    # Supported image extensions
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    def scan_and_remove_unlabeled(dataset_path, images_subdir="images", labels_subdir="labels"):
        """
        Scan dataset and remove images without labels
        """
        images_path = os.path.join(dataset_path, images_subdir)
        labels_path = os.path.join(dataset_path, labels_subdir)
        
        if not os.path.exists(images_path):
            print(f"❌ Images directory not found: {images_path}")
            return
        
        if not os.path.exists(labels_path):
            print(f"❌ Labels directory not found: {labels_path}")
            return
        
        print(f"🔍 Scanning dataset: {dataset_path}")
        print(f"📁 Images: {images_path}")
        print(f"🏷️  Labels: {labels_path}")
        
        # Get all image files
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(images_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(images_path, f"*{ext.upper()}")))
        
        print(f"📊 Found {len(image_files)} image files")
        
        removed_count = 0
        kept_count = 0
        
        for image_file in image_files:
            # Get corresponding label file
            image_name = Path(image_file).stem
            label_file = os.path.join(labels_path, f"{image_name}.txt")
            
            should_remove = False
            reason = ""
            
            # Check if label file exists
            if not os.path.exists(label_file):
                should_remove = True
                reason = "No label file"
            else:
                # Check if label file is empty or has no valid annotations
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if not content:
                            should_remove = True
                            reason = "Empty label file"
                        else:
                            # Check if all lines are valid YOLO format
                            lines = content.split('\n')
                            valid_lines = 0
                            for line in lines:
                                line = line.strip()
                                if line:  # Skip empty lines
                                    parts = line.split()
                                    if len(parts) == 5:  # class_id x_center y_center width height
                                        try:
                                            # Validate that all parts are numbers
                                            class_id = int(parts[0])
                                            coords = [float(x) for x in parts[1:]]
                                            valid_lines += 1
                                        except ValueError:
                                            continue
                            
                            if valid_lines == 0:
                                should_remove = True
                                reason = "No valid annotations"
                
                except Exception as e:
                    should_remove = True
                    reason = f"Error reading label file: {str(e)}"
            
            if should_remove:
                print(f"🗑️  Removing: {os.path.basename(image_file)} ({reason})")
                
                # Remove image file
                os.remove(image_file)
                
                # Remove label file if it exists
                if os.path.exists(label_file):
                    os.remove(label_file)
                
                removed_count += 1
            else:
                kept_count += 1
        
        print(f"\n✅ Cleanup complete!")
        print(f"🗑️  Removed: {removed_count} images (no defects)")
        print(f"✅ Kept: {kept_count} images (with defects)")
        print(f"📊 Dataset size reduced by: {removed_count/(removed_count+kept_count)*100:.1f}%")
        
        return removed_count, kept_count
    
    # Example usage for different dataset structures
    
    # Structure 1: Simple structure (images and labels in same root)
    # your_dataset/
    # ├── images/
    # └── labels/
    
    print("=" * 60)
    print("🧹 DATASET CLEANER - Remove Images Without Labels")
    print("=" * 60)
    
    # Demo with common dataset structures
    dataset_structures = [
        {"path": "imgwarp/warpeddata", "images": "images", "labels": "labels"},
        {"path": "dataset", "images": "images/train", "labels": "labels/train"},
        {"path": "dataset", "images": "images/val", "labels": "labels/val"},
        {"path": "yolo_dataset", "images": "train/images", "labels": "train/labels"},
        {"path": "yolo_dataset", "images": "val/images", "labels": "val/labels"},
    ]
    
    print("📋 Common dataset structures to check:")
    for i, struct in enumerate(dataset_structures):
        print(f"{i+1}. {struct['path']}/{struct['images']} + {struct['path']}/{struct['labels']}")
    
    print("\n" + "="*60)
    print("🔧 TO USE THIS SCRIPT:")
    print("1. Update DATASET_PATH to your actual dataset path")
    print("2. Update IMAGES_DIR and LABELS_DIR if needed")
    print("3. Run the function: scan_and_remove_unlabeled()")
    print("="*60)
    
    # Example function call (commented out - uncomment and modify for your use)
    # removed, kept = scan_and_remove_unlabeled("imgwarp/warpeddata", "images", "labels")
    
    return scan_and_remove_unlabeled

# Create the function
clean_function = clean_dataset_remove_unlabeled()
clean_function('/home/ahmet/work/yildizimgs_objdet/yolov5/imgwarp/warpeddata_clean/train')
clean_function('/home/ahmet/work/yildizimgs_objdet/yolov5/imgwarp/warpeddata_clean/valid')
# Show example of how to use it
#print("\n🚀 READY TO USE!")
#print("Example usage:")
#print("removed, kept = clean_function('your_dataset_path', 'images', 'labels')")
