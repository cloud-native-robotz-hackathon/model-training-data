import cv2
import argparse
import os

def draw_yolo_bounding_boxes(image_path, annotation_path, output_path=None):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img_height, img_width = image.shape[:2]

    # Read annotation file
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines

            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized values to pixel values
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Draw rectangle
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, str(int(class_id)), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show or save the image
    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("YOLO Annotations", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw YOLO bounding boxes on an image.")
    parser.add_argument("image", help="Path to the image file")
    parser.add_argument("annotation", help="Path to the YOLO annotation file (.txt)")
    parser.add_argument("-o", "--output", help="Path to save the output image", default=None)
    args = parser.parse_args()

    draw_yolo_bounding_boxes(args.image, args.annotation, args.output)

