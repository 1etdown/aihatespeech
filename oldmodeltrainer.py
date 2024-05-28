# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# import joblib
# from flask import Flask, request, jsonify
# import csv
#
# # Загрузка необходимых ресурсов NLTK
# nltk.download('stopwords')
# nltk.download('wordnet')
#
# # Функция предобработки текста
# def preprocess_text(text):
#     text = re.sub(r'<.*?>', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.lower()
#     tokens = text.split()
#     stop_words = set(stopwords.words('russian'))
#     tokens = [word for word in tokens if word not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     return ' '.join(tokens)
#
# # Загрузка данных из CSV с дополнительными параметрами
# data = pd.read_csv('final_data.csv', delimiter=';', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, on_bad_lines='warn')
#
# # Проверка первых строк данных для диагностики
# print(data.head())
#
# # Предположим, что столбец "text" содержит тексты комментариев, а "hate_speech" - метки классов
# X = data['text']
# y = data['hate_speech']
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Преобразование текста в TF-IDF признаки
# vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)
#
# # Обучение модели
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)
#
# # Сохранение модели и векторизатора
# joblib.dump(model, 'model.pkl')
# joblib.dump(vectorizer, 'vectorizer.pkl')
#
# # Предсказание и оценка модели
# y_pred = model.predict(X_test_tfidf)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
