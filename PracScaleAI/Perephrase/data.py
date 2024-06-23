import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Примеры текстов (добавлено больше примеров)
original_texts = [
    "Пример первого текста",
    "Пример второго текста",
    "Третий пример текста",
    "Четвертый пример текста для обучения",
    "Пятый пример текста для модели",
    "Этот текст предназначен для тестирования модели",
    "Модель должна уметь перефразировать этот текст",
    "Этот пример показывает, как работает модель",
    "Еще один пример текста для обучения",
    "Последний пример текста для модели",
    "Текст для обучения нейросети",
    "Пример использования нейросети для перефразирования"
]

paraphrased_texts = [
    "Перефразированный первый текст",
    "Перефразированный второй текст",
    "Перефразированный третий текст",
    "Перефразированный четвертый текст для обучения",
    "Перефразированный пятый текст для модели",
    "Этот текст служит для тестирования модели",
    "Модель должна быть способна перефразировать данный текст",
    "Этот пример демонстрирует работу модели",
    "Перефразированный еще один пример текста для обучения",
    "Перефразированный последний пример текста для модели",
    "Перефразированный текст для обучения нейросети",
    "Пример использования нейросети для перефразирования текста"
]
# Токенизация и создание последовательностей
tokenizer = Tokenizer()
tokenizer.fit_on_texts(original_texts + paraphrased_texts)
sequences_original = tokenizer.texts_to_sequences(original_texts)
sequences_paraphrased = tokenizer.texts_to_sequences(paraphrased_texts)

# Паддинг последовательностей
max_sequence_len = max(max(len(x) for x in sequences_original), max(len(x) for x in sequences_paraphrased))
padded_original = pad_sequences(sequences_original, maxlen=max_sequence_len, padding='post')
padded_paraphrased = pad_sequences(sequences_paraphrased, maxlen=max_sequence_len, padding='post')

# Размер словаря
vocab_size = len(tokenizer.word_index) + 1

# Параметры модели
embedding_dim = 256
lstm_units = 128

# Архитектура Seq2Seq модели
input_seq = Input(shape=(max_sequence_len,))
embedded_seq = Embedding(vocab_size, embedding_dim)(input_seq)
encoder = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedded_seq)
decoder = LSTM(lstm_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(encoder)
decoder_dense = Dense(vocab_size, activation='softmax')
output = decoder_dense(decoder_outputs)

model = Model(input_seq, output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Увеличиваем количество эпох
model.fit(padded_original, padded_paraphrased, batch_size=64, epochs=50)

# Функция для перефразирования текста
def paraphrase(text):
    sequence = tokenizer.texts_to_sequences([text])
    print(f"Tokenized sequence: {sequence}")  # Отладочный вывод
    if not sequence[0]:  # Проверка на пустую последовательность
        return "Токены не найдены для входного текста."
    
    padded = pad_sequences(sequence, maxlen=max_sequence_len, padding='post')
    print(f"Padded sequence: {padded}")  # Отладочный вывод
    prediction = model.predict(padded)
    print(f"Prediction: {prediction}")  # Отладочный вывод
    predicted_sequence = np.argmax(prediction, axis=-1)[0]  # Получаем одномерный массив
    print(f"Predicted sequence: {predicted_sequence}")  # Отладочный вывод
    paraphrased_text = tokenizer.sequences_to_texts([predicted_sequence])[0]
    return paraphrased_text

# Пример использования функции
result = paraphrase("")
print(f"Paraphrased text: {result}")