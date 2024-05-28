import tkinter as tk
from tkinter import messagebox
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Функция предобработки текста
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Загрузка модели и векторизатора
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Функция для предсказания и отображения результата
def predict_hate_speech():
    text = text_entry.get("1.0", tk.END).strip()
    if text:
        preprocessed_text = preprocess_text(text)
        text_tfidf = vectorizer.transform([preprocessed_text])
        prediction = model.predict(text_tfidf)[0]
        result = "Это хейт спич" if prediction == 1 else "Это не хейт спич"
        messagebox.showinfo("Результат", result)
    else:
        messagebox.showwarning("Ошибка ввода", "Вы ничего не ввели.")

# Создание интерфейса Tkinter
root = tk.Tk()
root.title("Hate Speech Детектор")

# Настройка виджетов
text_label = tk.Label(root, text="Введите сообщение:")
text_label.pack()

text_entry = tk.Text(root, height=10, width=50)
text_entry.pack()

predict_button = tk.Button(root, text="Определить", command=predict_hate_speech)
predict_button.pack()

# Запуск главного цикла Tkinter
root.mainloop()
