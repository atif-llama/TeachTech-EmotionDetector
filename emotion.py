from google.colab import drive
import os
import pandas as pd

drive.mount('/content/drive', force_remount = True)
CSV_PATH = '/content/drive/MyDrive/emotions.csv'
assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"


df_header = pd.read_csv(CSV_PATH, nrows = 0)
print("CSV Columns:", df_header.columns.tolist())


import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

CHUNKSIZE = 1000
TEXT_COL = 'text'
N_FEATURES = 2**18
LABEL_COL = 'label'
MODEL_PATH = '/content/drive/MyDrive/emotion_clf.pkl'


def train_and_save(csv_path, model_path):
  header = pd.read_csv(csv_path, nrows = 0).columns.tolist()
  if TEXT_COL not in header or LABEL_COL not in header:
    raise KeyError (f"Required columns not found. Available: {header}")

    vectorizer = HashingVectorizer(
    n_feature = N_FEATURES,
    norm = None,
    binary = False
    )
    encoder = LabelEncoder
    classifier = SGDClassifier(
        loss = 'log_loss',
        max_iter = 1,
        tol = None,
        learning_rate = 'optimal',
        random_state = 42
    )
    first_pass = True
    classes = None
    for chunk in pd.read_csv(csv_path, chunksize=CHUNKSIZE):
      texts = chunk[TEXT_COL].astype(str).tolist()
      x = vectorizer.transform(texts)
      y_raw = chunk[LABEL_COL].values

      if first_pass:
        encoder.fit(y_raw)
        classes.encoder.transform(encoder.classes_)
        first_pass = False

      y = encoder.transform(y_raw)
      if not hasattr(classifier, 'classes_'):
        classifier.partial_fit(x,y)
      else:
        classifier.partial_fit(x,y)

    joblib.dump({'model': classifier, 'vectorizer': vectorizer, 'encoder': encoder}, model_path)
    print(f"Training complete. Model saved to {model_path}")

def load_and_predict(text,model_path):
    data = joblib.load(model_path)
    model = data['model']
    vectorizer = data['vectorizer']
    encoder = data['encoder']

    x_new = vectorizer.transform([text])
    y_pred = model.predict(x_new)
    return encoder.inverse_transform(y_pred)[0]

train_and_save = (CSV_PATH, MODEL_PATH)
sample_text = "I feel happy and energetic today"
predicted_emotion = load_and_predict(sample_text, MODEL_PATH)
print(f"Input: {sample_text}\nPredicted Emotion: {predicted_emotion}")
