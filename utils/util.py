import numpy as np
import cv2
import tensorflow as tf


def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    Each box is given in the format [xmin, ymin, xmax, ymax].
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the Intersection over Union (IoU)
    iou_value = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou_value


def nm_suppression(bboxes, class_indices, confidences, iou_threshold=0.5, conf=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
    """
    indices = np.argsort(confidences)[::-1]  # Sort boxes by confidence score in descending order
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]

        # Suppress boxes with high IoU overlap
        filtered_indices = []
        for idx in indices:
            if iou(bboxes[current], bboxes[idx]) < iou_threshold and confidences[idx] >= conf:
                filtered_indices.append(idx)
        indices = np.array(filtered_indices)

    # Select boxes that are kept
    return bboxes[keep], [class_indices[i] for i in keep], [confidences[i] for i in keep]


def non_maximum_suppression(bboxes, class_indices, confidences, iou_threshold=0.5, conf=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.
    """
    indices = np.argsort(confidences)[::-1]  # Sort boxes by confidence score in descending order
    class_index = []
    for clas_ind in class_indices:
        class_index.append(np.argmax(clas_ind))
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]

        # Suppress boxes with high IoU overlap
        filtered_indices = []
        for idx in indices:
            if iou(bboxes[current], bboxes[idx]) < iou_threshold and confidences[idx] >= conf:
                filtered_indices.append(idx)
        indices = np.array(filtered_indices)

    # Select boxes that are kept
    return bboxes[keep], [class_index[i] for i in keep], [confidences[i] for i in keep]


def remove_border(image):
    inv_image = np.bitwise_not(image)
    contours, _ = cv2.findContours(inv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Iterate through the contours
    for contour in contours:
        # Check if the contour touches the image boundaries
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x + w == width or y + h == height:
            # Fill the contour with white color
            cv2.drawContours(image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return image


def three_eye_detect(image_path, model_path, input_size=(224, 224)):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)  #If passing image file path
    else:
        img = image_path  # If passing cv2 image

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_image = cv2.resize(image_rgb, input_size)
    input_image = input_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Process output data
    # Adjust according to your model's output structure (YOLOv5 typically has 3 output arrays)
    bboxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding boxes

    # Scale bounding boxes to original image dimensions
    h, w, _ = img.shape
    bboxes[:, [0, 2]] *= w  # Scale x-coordinates
    bboxes[:, [1, 3]] *= h  # Scale y-coordinates

    # Filter out detections with low confidence
    threshold = 0.5  # Set a confidence threshold

    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # xmin
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # ymin
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # xmax
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # ymax
    bboxes[:, 5] = bboxes[:, 5].astype(int)
    filtered_bboxes, filtered_class_indices, filtered_confidences = nm_suppression(
        bboxes[:, :4], bboxes[:, 5], bboxes[:, 4], iou_threshold=0.5, conf=0.75
    )

    if len(filtered_confidences) >= 2:

        return True
    else:
        return False


def is_blurry_image(input_image):
    try:
        # Convert to grayscale if the image has multiple channels
        if len(input_image.shape) > 2:
            gMat1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gMat1 = input_image.copy()

        # Apply Laplacian transformation
        laplacian = cv2.Laplacian(gMat1, cv2.CV_64F)

        # Calculate mean and standard deviation of the Laplacian
        mean, stddev = cv2.meanStdDev(laplacian)

        # Get the variance (square of the standard deviation)
        variance = stddev[0][0] ** 2

        # Check if the image is blurry based on variance threshold
        if variance >= 9:  # Adjust the threshold as needed
            return False  # Not blurry
        else:
            return True  # Blurry
    except Exception as e:
        return True


# Placeholder values for constants, replace with actual values
TF_OD_API_INPUT_SIZE_DIGITALUNISIGN_YOLOV5 = 416  # Example size, adjust as needed
MINIMUM_ORIGINAL_CONFIDENCE_TF_OD_API = 0.8  # Confidence threshold for YOLOv5


def detection_tflite(image_path, model_path, input_size=(416, 416)):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)  #If passing image file path
    else:
        image = image_path  # If passing cv2 image
    if len(image.shape) >= 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    input_image = cv2.resize(image_rgb, input_size)
    input_image = input_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Process output data
    # Adjust according to your model's output structure (YOLOv5 typically has 3 output arrays)
    bboxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding boxes

    # Scale bounding boxes to original image dimensions
    h, w = input_image.shape[:2]
    bboxes[:, [0, 2]] *= 416  # Scale x-coordinates
    bboxes[:, [1, 3]] *= 416  # Scale y-coordinates

    # Filter out detections with low confidence
    threshold = 0.5  # Set a confidence threshold
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # xmin
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # ymin
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # xmax
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # ymax
    # bboxes[:, 5] = bboxes[:, 5].astype(int)
    filtered_bboxes, filtered_class_indices, filtered_confidences = non_maximum_suppression(
        bboxes[:, :4], bboxes[:, 5:], bboxes[:, 4], iou_threshold=0.6, conf=0.55
    )

    lst_result = []
    cls_names = ['bottom', 'dot', 'left']
    for conf, cls, xy in zip(filtered_confidences, filtered_class_indices, filtered_bboxes):
        score = conf
        location = xy

        # Skip processing if the location is None or the confidence score is too low
        if location is None or score < 0.55:
            continue

        title = cls_names[cls]

        # Calculate initial ROI
        roi_inside = {
            'x': max(0, int(location[0] * 540 / 416)),
            'y': max(0, int(location[1] * 720 / 416)),
            'width': min(int((location[2] - location[0]) * 540 / 416), 540),
            'height': min(int((location[3] - location[1]) * 720 / 416), 720)
        }

        roi = {
            'x': max(0, roi_inside['x'] - 50),
            'y': max(0, roi_inside['y'] - 20),
            'width': min(roi_inside['width'] + 120, 540 - roi_inside['x']),
            'height': min(roi_inside['height'] + 60, 720 - roi_inside['y'])
        }

        try:
            if 'left' in title and score >= 0.55:
                # Process 'left' titled results
                new_roi = roi_inside
                final_mat = image_gray[new_roi['y']:new_roi['y'] + new_roi['height'],
                            new_roi['x']:new_roi['x'] + new_roi['width']]
                lst_result.append(final_mat)

            elif 'bottom' in title and score >= 0.55:
                # Adjusting ROI for 'bottom' titled results
                roi['width'] += 50
                roi['height'] += 5
                new_roi = roi
                final_mat = image_gray[new_roi['y']:new_roi['y'] + new_roi['height'],
                            new_roi['x']:new_roi['x'] + new_roi['width']]
                lst_result.append(final_mat)

            elif 'dot' in title and score >= 0.55:
                # Process 'dot' titled results
                new_roi = roi_inside
                final_mat = image_gray[new_roi['y']:new_roi['y'] + new_roi['height'],
                            new_roi['x']:new_roi['x'] + new_roi['width']]
                lst_result.append(final_mat)

        except Exception as e:
            continue

    return lst_result


def process_dots(image_path, model_path, input_size=(416, 416)):  #, results, inp, tf_input_size, min_original_conf=0.8, min_fake_conf=0.5):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)  # If passing image file path
    else:
        image = image_path  # If passing cv2 image
    if len(image.shape) >= 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image
    image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    input_image = cv2.resize(image_rgb, input_size)
    input_image = input_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_image)

    # Run inference
    interpreter.invoke()

    # Process output data
    # Adjust according to your model's output structure (YOLOv5 typically has 3 output arrays)
    bboxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding boxes

    # Scale bounding boxes to original image dimensions
    h, w = input_image.shape[:2]
    bboxes[:, [0, 2]] *= 416  # Scale x-coordinates
    bboxes[:, [1, 3]] *= 416  # Scale y-coordinates

    # Filter out detections with low confidence
    threshold = 0.5  # Set a confidence threshold
    bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # xmin
    bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # ymin
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # xmax
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # ymax

    filtered_bboxes, filtered_class_indices, filtered_confidences = non_maximum_suppression(
        bboxes[:, :4], bboxes[:, 5:], bboxes[:, 4], iou_threshold=0.6, conf=0.50
    )

    cls_names = ["Fake_Dot", "Fake_Left", "Original_Dot", "Original_Left"]
    min_original_conf = 0.8
    min_fake_conf = 0.5
    org_counter = 0
    fake_counter = 0

    for conf, cls, xy in zip(filtered_confidences, filtered_class_indices, filtered_bboxes):
        location = xy
        confidence = conf

        if location is not None:
            # Check for "Original_Dot" with the required confidence
            if confidence >= min_original_conf and "Original_Dot" in cls_names[cls]:
                # Calculate the bounding box (ROI)
                roi_inside = {
                    'x': max(0, int(location[0] * 540 / 416)),
                    'y': max(0, int(location[1] * 720 / 416)),
                    'width': min(int((location[2] - location[0]) * 540 / 416), 540),
                    'height': min(int((location[3] - location[1]) * 720 / 416), 720)
                }

                # Increment original counter
                org_counter += 1


            # Check for "Fake_Dot" with the required confidence
            elif confidence >= min_fake_conf and "Fake_Dot" in cls_names[cls]:
                # Calculate the bounding box (ROI)
                roi_inside = {
                    'x': max(0, int(location[0] * 540 / 416)),
                    'y': max(0, int(location[1] * 720 / 416)),
                    'width': min(int((location[2] - location[0]) * 540 / 416), 540),
                    'height': min(int((location[3] - location[1]) * 720 / 416), 720)
                }

                # Increment fake counter
                fake_counter += 1

    if org_counter >= fake_counter:
        return True
    else:
        return False
    # return org_counter, fake_counter


if __name__ == "__main__":
    # Usage example
    model_path = '../models/dot_416_gray_126092024.tflite'  # Path to your TFLite model
    image_path = '../data/1.png'  # Path to the image you want to test

    detections = process_dots(image_path, model_path)