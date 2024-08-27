from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Define the ArcFace layer
class ArcFace(tf.keras.layers.Layer):
    def __init__(self, n_classes, s=30.0, m=0.5, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.W = self.add_weight(name='W', shape=(256, n_classes), initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = tf.matmul(x, W)
        theta = tf.acos(tf.clip_by_value(logits, -1.0, 1.0))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        return logits

# Load the trained model
model_path = 'face_recognition_transfer_model.keras'
transfer_model = load_model(model_path, custom_objects={'ArcFace': ArcFace})

# Load the base model used for generating embeddings
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')

# Load label encoder to decode the predicted labels
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

app = Flask(__name__)
CORS(app)

@app.route('/recognize', methods=['POST'])
def recognize():
    logging.debug('Received request for recognition')

    # Get the image from the request
    file = request.files['image']
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Use MTCNN for face detection
    detector = MTCNN()
    faces = detector.detect_faces(img)
    logging.debug(f'Detected faces: {faces}')

    response = []

    for face in faces:
        x, y, w, h = face['box']
        # Extract the face ROI
        face_img = img[y:y+h, x:x+w]

        # Preprocess the face image
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32')
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        # Generate embedding
        face_embedding = base_model.predict(face_img)[0]

        # Prepare the inputs for the transfer model
        y_dummy = np.zeros((1, len(label_encoder.classes_)))

        # Predict the identity
        predictions = transfer_model.predict([np.expand_dims(face_embedding, axis=0), y_dummy])

        # Get the predicted class and its confidence
        predicted_label = np.argmax(predictions, axis=1)
        confidence_scores = predictions[0]
        confidence = confidence_scores[predicted_label[0]]

        if confidence >= 0.9:
            predicted_name = label_encoder.inverse_transform(predicted_label)[0]
        else:
            predicted_name = 'Unknown'

        response.append({
            'name': predicted_name,
            'confidence': float(confidence)
        })

    logging.debug(f'Response: {response}')
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
