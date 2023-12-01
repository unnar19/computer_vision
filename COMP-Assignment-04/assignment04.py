import cv2
import numpy as np
import time

def pre_process(input_image, net):
    net.setInput(cv2.dnn.blobFromImage(input_image, 1/255, (640, 640), [0,0,0], 1, crop=False))
    
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    return outputs


def post_process(input_image, outputs):
    class_ids = []
    confidences = []
    boxes = []

    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        if confidence >= 0.45:
            classes_scores = row[5:]

            class_id = np.argmax(classes_scores)

            if (classes_scores[class_id] > 0.5):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
              
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.45)
    
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        cv2.rectangle(input_image, (left,  top), (left+width, top+height), (0,255,0), 1)

        predicted_class = str(classes[class_ids[i]]).capitalize()
        prediction_confidence = confidences[i]
        print(classes[class_ids[i]])
        cv2.putText(input_image, predicted_class, (left+ 10, top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(input_image, f"{prediction_confidence:.2f}", (left+ 10, top + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    print()

    return input_image

prev_time = time.time()
keep_delta = []
print_delta = 0
print_fps = 0
    
cap = cv2.VideoCapture(0)
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


while(True):
    ret, frame = cap.read()

    modelWeights = "models/yolov5m.onnx"
    net = cv2.dnn.readNet(modelWeights)

    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)

    time_delta = time.time() - prev_time
    prev_time = time.time()
    keep_delta.append(time_delta)
    if len(keep_delta) == 5:
        print_delta = np.mean(keep_delta) * 1000
        print_fps = 1/np.mean(keep_delta)

        keep_delta, keep_score = [], []

    cv2.putText(img, f"{'FPS: '} {print_fps:.2f}",(8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)
    cv2.putText(img, f"{'Delay:'} {print_delta:.0f} ms",(8,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0),1, cv2.LINE_AA)

    cv2.imshow('YOLOv5m', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break