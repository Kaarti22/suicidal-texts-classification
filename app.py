from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("all")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

model = tf.keras.models.load_model("suicide_detection_lstm.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

app = FastAPI()

class TextInput(BaseModel):
    text: str

max_len = 200

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\W", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    return ""

@app.post("/predict/")
async def predict(input: TextInput):
    clean_text = preprocess_text(input.text)

    text_sequence = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_len, padding='post', truncating='post')

    prediction = model.predict(padded_sequence)[0][0]
    result = "Suicidal" if prediction > 0.5 else "Non-Suicidal"

    return {"prediction": result, "confidence": float(prediction)}