from flask import Flask, request, render_template, jsonify
import sys
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import pickle
from keras.models import load_model

app = Flask(__name__)

model_path = 'AI_vs_Human.h5'
model = load_model(model_path)

tokenizer_path = 'D:\\SudyML\\AIscale\\app\\tokenizer1.pickle'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    
    prediction = model.predict(padded)
    
    ai_probability = prediction[0][0]
    
    return ai_probability

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.form['text']
    if user_input.strip():
        confidence = predict_text(user_input) * 100
        return jsonify({'result': f'The text is {confidence:.2f}% AI-generated.'})
    else:
        return jsonify({'result': 'Please enter some text to analyze.'})

if __name__ == '__main__':
    app.run(debug=True)
