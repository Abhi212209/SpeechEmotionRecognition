import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, redirect, url_for
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import glob

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emotion labels mapping
emotions = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3}

# Feature extraction function
def extract_features(file_name):
    y, sr = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

# Prepare training data
X, y = [], []
for emotion, label in emotions.items():
    for file in glob.glob(f'data/{emotion}/*.wav'):
        try:
            features = extract_features(file)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {file}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Model training
model = SVC(kernel='linear')
model.fit(X, y)

# Save the trained model and scaler
pickle.dump(model, open('emotion_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Extract features from uploaded file
    features = extract_features(file_path).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)[0]

    # Map predicted label to emotion
    predicted_emotion = list(emotions.keys())[list(emotions.values()).index(prediction)]
    return f"Predicted Emotion: {predicted_emotion}"

if __name__ == '__main__':
    app.run(debug=True)
