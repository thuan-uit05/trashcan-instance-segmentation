import numpy as np
import cv2
import tensorflow as tf
tflite = tf.lite

class Model(object):
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.5):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]
        self.floating_model = self.input_details[0]["dtype"] == np.float32

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def _iou(self, box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _nms(self, boxes, scores):
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            indices = [i for i in indices if self._iou(boxes[current], boxes[i]) < self.iou_thresh]
        return keep

    def prepare(self):
        return None

    def predict(self, image):
        if hasattr(image, 'mode'):
            image = np.array(image.convert("RGB"))

        h_orig, w_orig = image.shape[:2]
        resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
        if resized.shape[2] == 1:
            resized = np.repeat(resized, 3, axis=2)

        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        proto = self.interpreter.get_tensor(self.output_details[1]["index"])[0]
        proto = cv2.resize(proto, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        boxes = output[:, 0:4]
        scores = output[:, 4]
        class_ids = output[:, 5].astype(int)

        converted_boxes = [
            [x1 * w_orig, y1 * h_orig, x2 * w_orig, y2 * h_orig]
            for x1, y1, x2, y2 in boxes
        ]

        filtered_idxs = [i for i, score in enumerate(scores) if score >= self.conf_thresh]
        filtered_boxes = [converted_boxes[i] for i in filtered_idxs]
        filtered_scores = [scores[i] for i in filtered_idxs]
        filtered_classes = [class_ids[i] for i in filtered_idxs]
        filtered_coeffs = [output[i, 6:38] for i in filtered_idxs]

        nms_indices = self._nms(filtered_boxes, filtered_scores)

        results = []
        for i in nms_indices:
            x1, y1, x2, y2 = [int(c) for c in filtered_boxes[i]]
            class_id = filtered_classes[i]
            coeffs = filtered_coeffs[i]

            mask = np.tensordot(proto, coeffs, axes=([2], [0]))
            mask = 1 / (1 + np.exp(-mask))
            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            mask_crop = binary_mask[y1:y2, x1:x2]
            contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                polygon = largest.reshape(-1, 2) + np.array([x1, y1])
                polygon_coords = polygon.flatten().tolist()

                if len(polygon_coords) % 2 != 0:
                    polygon_coords = polygon_coords[:-1]

                result = [class_id, float(filtered_scores[i])] + polygon_coords
                results.append(result)


        return results