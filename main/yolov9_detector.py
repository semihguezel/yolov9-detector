#! /usr/bin/env python3

# Import built-in libraries
import os
import time
import threading

# Import third-party packages
import cv2
import torch
import numpy as np
from PIL import Image

# Import yolo-v9 helper modules
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode

# Import custom modules

from helpers.frame_queue_handler import FrameQueueHandler
from helpers.utilities import generate_random_rgb_color_list, read_yaml_file, get_labels_from_txt


class YoloDetector:
    def __init__(self):
        """
        Initialize the YoloDetector class.

        This class is designed for object detection using YOLO (You Only Look Once) model.

        Attributes:
        - video_path (str): The path to the input video file.
        - frame_rate (int): The frame rate at which frames are processed.
        - model (DetectMultiBackend): The YOLO-v9 model for object detection.
        - labels (dict): Detection classes are represented in key-value pairs.
        - image_size (int): The image size will be utilized during the model's prediction process.
        - confidence (float): Confidence threshold for object detection.
        - iou (float): Intersection over Union (IOU) threshold for object detection.
        - max_detection (int): The maximum number of detectable objects in a single frame.
        - font (int): OpenCV font for displaying class names.
        - font_scale (float): Font scale for displaying class names.
        - line_thickness (int): Thickness of lines in bounding boxes.
        - num_classes (int): Number of classes in the YOLO-v9 model.
        - color_list (list): List of random RGB colors for different classes.
        - queue_handler (FrameQueueHandler): Queue handler for managing frames in a thread-safe manner.
        """

        parent_directory = os.path.dirname(os.getcwd())
        self.parsed_data = read_yaml_file(parent_directory + "/config/configuration.yaml")

        self.video_path = self.parsed_data['video_path']
        self.frame_rate = self.parsed_data['frame_rate']

        # Model Parameters
        self.device = select_device(torch.cuda.current_device())
        self.model = DetectMultiBackend(
            parent_directory + self.parsed_data['model_parameters']['model_path'],
            device=self.device,
            fp16=False
        )
        self.labels = get_labels_from_txt(parent_directory + self.parsed_data['model_parameters']['label_path'])
        self.image_size = self.parsed_data['model_parameters']['image_size']

        # Parameters for NMS algorithm
        self.confidence = self.parsed_data['model_parameters']['conf']
        self.iou = self.parsed_data['model_parameters']['iou']
        self.max_detection = self.parsed_data['model_parameters']['max_detection']

        # Parameters for drawing purposes
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.font_scale = 1
        self.line_thickness = 2
        self.num_classes = len(self.model.names)
        self.color_list = generate_random_rgb_color_list(self.num_classes)

        # Initialize queue handler for video frames.
        self.queue_handler = FrameQueueHandler(max_size=5)

    def read_frames(self):
        """
        Read frames from the input video and put them in the queue.

        This method reads frames from the input video and puts them into the frame queue for further processing.
        """

        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            self.queue_handler.put(frame)
            time.sleep(1 / self.frame_rate)

        cap.release()

        cv2.destroyAllWindows()

    @smart_inference_mode()
    def predict(self):
        """
        Perform object detection on the next frame from the queue.

        Returns:
            numpy.ndarray: The frame with bounding boxes and class labels drawn.
        """
        frame = self.queue_handler.get()

        if frame is not None:
            # Load image
            color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            img0 = np.array(pil_image)
            img = letterbox(img0, self.image_size, stride=self.model.stride, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device).float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            predictions = self.model(img, augment=False, visualize=False)

            # Apply NMS
            predictions = non_max_suppression(
                predictions[0][0],
                self.confidence,
                self.iou,
                classes=None,
                max_det=self.max_detection
            )

            # The inference section is largely adapted from the provided example within the YOLO-v9 repository.
            # For a more detailed examination, feel free to explore the Google Colab page associated with the implementation.
            # https://colab.research.google.com/drive/1U3rbOmAZOwPUekcvpQS4GGVJQYR7VaQX?usp=sharing#scrollTo=u9ZdA2xx27nl

            # Process detections
            for detections in predictions:
                if len(detections):
                    detections[:, :4] = scale_boxes(img.shape[2:], detections[:, :4], img0.shape).round()
                    for *xyxy, conf, cls in reversed(detections):
                        xyxy = torch.stack(xyxy).cpu().numpy().reshape(1, -1)
                        x1 = int(xyxy[0][0])
                        y1 = int(xyxy[0][1])
                        x2 = int(xyxy[0][2])
                        y2 = int(xyxy[0][3])

                        class_name = self.labels[int(cls)]
                        confidence_score = round(float(conf) * 100, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                        cv2.putText(frame, '{} [{}]'.format(class_name, confidence_score),
                                    (x1, y1), 1, 1,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA)

            return frame

    def process_and_display_frames(self):
        """
        Continuously process frames and display the result.

        This method continuously processes frames using the inference method and displays the resulting frames.
        """
        while True:
            processed_frame = self.predict()

            if processed_frame is not None:
                cv2.imshow('Frame', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Instantiate YoloDetector
    yolo_detector = YoloDetector()

    # Create a thread to read frames from the video
    read_thread = threading.Thread(target=yolo_detector.read_frames)
    read_thread.start()

    # Process and display frames in the main thread
    yolo_detector.process_and_display_frames()
