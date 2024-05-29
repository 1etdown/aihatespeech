import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import TextClassificationPipeline
import torch
import joblib
from flask import Flask, request, jsonify
import csv

# Загрузка необходимых ресурсов NLTK
nltk.download('stopwords')

# Функция предобработки текста
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    stop_words = set(stopwords.words('russian'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Загрузка данных из CSV с дополнительными параметрами
data = pd.read_csv('final_data.csv', delimiter=';', quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL, on_bad_lines='warn')

# Проверка первых строк данных для диагностики
print(data.head())

# Предположим, что столбец "text" содержит тексты комментариев, а "hate_speech" - метки классов
X = data['text'].apply(preprocess_text)
y = data['hate_speech']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация токенизатора и модели BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

# Создание датасета для тренировки
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Параметры тренировки
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Создание тренировочного и тестового датасетов
train_dataset = CustomDataset(X_train.to_list(), y_train.to_list(), tokenizer, max_len=128)
test_dataset = CustomDataset(X_test.to_list(), y_test.to_list(), tokenizer, max_len=128)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Тренировка модели
trainer.train()

# Оценка модели
trainer.evaluate()

# Сохранение модели и токенизатора
model.save_pretrained('bert_model')
tokenizer.save_pretrained('bert_tokenizer')

