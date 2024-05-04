from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load class labels
classFile = '/Users/useradmin/Documents/object_detection/coco_class_labels.txt'
with open(classFile) as fp:
    labels = fp.read().split("\n")

# Load pre-trained model
modelFile = '/Users/useradmin/Documents/object_detection/frozen_inference_graph.pb'
configFile = '/Users/useradmin/Documents/object_detection/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'

net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

# Function to detect objects
def detect_objects(im, dim=300):
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=127.5, swapRB=True, crop=False)
    net.setInput(blob)
    objects = net.forward()
    return objects

# Function to display objects
def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]
    for i in range(objects.shape[2]):
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)
        if score > threshold:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(im, "{}".format(labels[classId]), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    return im

# API endpoint for object detection
@app.route('/detect_objects', methods=['POST'])
def detect_objects_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    nparr = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    objects = detect_objects(img)
    result_image = display_objects(img, objects)
    
    retval, buffer = cv2.imencode('.jpg', result_image)
    img_str = buffer.tobytes()
    
    return img_str, 200, {'Content-Type': 'image/jpeg'}

if __name__ == '__main__':
    app.run(debug=True)
