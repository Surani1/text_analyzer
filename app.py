import nltk

nltk.download('punkt')
nltk.download('stopwords')

import os
import base64
import random
import re
import socket
import logging
import logging.config
from datetime import datetime
from collections import Counter
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok
import docx
from docx import Document
import language_tool_python
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import statistics as st
import yaml
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import yake
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Инициализация Flask приложения
app = Flask(__name__)

# Загрузка конфигурации логирования
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Загрузка конфигурации логирования
with open('config/logging.yaml', 'r', encoding='utf-8') as f:
    logging_config = yaml.safe_load(f)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
logger.debug("Программа запущена успешно. Начало сбора логов")

# Константы
TRAINING_DATA = {
    'text': [
        'Я очень рад этому событию!', 'Это просто ужасно и невыносимо.',
        'Мне все равно на это.', 'Замечательный день для прогулки!',
        'Это было отвратительно.', 'Я счастлив быть здесь.',
        'Жизнь прекрасна.', 'Это было ужасно.',
        'Событие прошло успешно.', 'Я разочарован сегодня.', 'Сегодня отличный день!', 'Мне грустно.'
    ],
    'sentiment': [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
}

KEYWORD_EXTRACTOR_PARAMS = {"lan": "ru", "n": 1, "dedupLim": 0.3, "top": 5}
STOP_WORDS = stopwords.words('russian')

class TextProcessor:
    def __init__(self):
        self.preprocessor = lambda text: ' '.join(
            word for word in re.findall(r'\b[a-zA-Zа-яА-Я]+\b', text.lower())
            if word not in STOP_WORDS
        )
        self.model = self._train_model()
        self.keyword_extractor = yake.KeywordExtractor(**KEYWORD_EXTRACTOR_PARAMS)
        self.language_tool = language_tool_python.LanguageTool('ru')

    def _train_model(self):
        logger.info("Начало обучения модели")
        df = pd.DataFrame(TRAINING_DATA)
        df['processed_text'] = df['text'].apply(self.preprocessor)
        X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, random_state=42)
        logger.info(f"Размер обучающей выборки: {len(X_train)}, размер тестовой выборки: {len(X_test)}")

        model = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', LogisticRegression())])
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        logger.info(f"Точность модели: {accuracy:.2f}")
        logger.info(f"Отчет о классификации:\n{classification_report(y_test, predictions)}")
        return model

    def predict_sentiment(self, text):
        processed = self.preprocessor(text)
        prediction = self.model.predict([processed])[0]
        result = "положительное" if prediction == 1 else "отрицательное"
        logger.info(f"Предсказанное настроение текста: {result}")
        return result

    def word_frequencies(self, text):
        words = self.preprocessor(text).split()
        return Counter(words)

    def analyze_ngrams(self, text, n=2):
        vectorizer = CountVectorizer(ngram_range=(n, n))
        ngrams = vectorizer.fit_transform([self.preprocessor(text)])
        return vectorizer.get_feature_names_out(), ngrams.toarray()

    def pos_tagging(self, text):
        words = nltk.word_tokenize(text)
        return nltk.pos_tag(words, lang='rus')

    def grammar_check(self, text):
        logger.info("Начало проверки грамматики")
        matches = self.language_tool.check(text)
        corrected_text = self.language_tool.correct(text)
        logger.info(f"Найдено {len(matches)} грамматических ошибок")
        return corrected_text, len(matches)

    def lexical_diversity(self, text):
        words = self.preprocessor(text).split()
        return len(set(words)) / len(words) if words else 0

    def text_complexity(self, text):
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        syllables = sum(self.count_syllables(word) for word in words)
        return 206.835 - 1.015 * (len(words) / len(sentences)) - 84.6 * (syllables / len(words))

    def count_syllables(self, word):
        return len(re.findall(r'[аеёиоуыэюя]', word.lower()))

    def plot_word_frequencies(self, text):
        freqs = self.word_frequencies(text)
        top_10_words = dict(sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:10])
        keywords = self.keyword_extractor.extract_keywords(text)
        keyword_words = [word.lower() for word, _ in keywords[:10]]

        plt.figure(figsize=(8, 14))

        # График частоты слов (верхний)
        plt.subplot(3, 1, 1)
        plt.bar(range(len(top_10_words)), list(top_10_words.values()), color='skyblue')
        plt.xticks(range(len(top_10_words)), list(top_10_words.keys()), rotation=45, ha='right', fontsize=10)
        plt.title('Топ-10 слов по частоте', fontsize=14)
        plt.ylabel('Частота', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # График значимости ключевых слов (средний)
        plt.subplot(3, 1, 2)
        keyword_scores = [score for _, score in keywords[:10]]
        plt.bar(range(len(keyword_scores)), keyword_scores, color='lightgreen')
        plt.xticks(range(len(keyword_scores)), keyword_words, rotation=45, ha='right', fontsize=10)
        plt.title('Значимость ключевых слов', fontsize=14)
        plt.ylabel('Значимость', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # График длины слов (нижний)
        plt.subplot(3, 1, 3)
        words = self.preprocessor(text).split()
        word_lengths = [len(word) for word in words]
        plt.hist(word_lengths, bins=range(1, max(word_lengths) + 2), color='orange', edgecolor='black', alpha=0.7)
        plt.title('Длина слов', fontsize=14)
        plt.xlabel('Длина слов', fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Настройка шрифтов и отступов
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.6)

        # Сохранение графика
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def analyze_text(self, text):
        logger.info("Начало анализа текста")
        processed_text = self.preprocessor(text)
        words = processed_text.split()
        word_lengths = [len(word) for word in words]

        corrected_text, num_errors = self.grammar_check(text)
        lexical_div = self.lexical_diversity(text)
        keywords = self.keyword_extractor.extract_keywords(text)

        keyword_words = [word.lower() for word, _ in keywords[:10]]

        logger.info(f"Количество слов в тексте: {len(words)}")
        logger.info(f"Лексическое разнообразие: {lexical_div:.2f}")
        logger.info(f"Извлечено ключевых слов: {len(keywords)}")

        result = {
            'Настроение': self.predict_sentiment(text),
            'Количество слов': len(words),
            'Количество символов': sum(word_lengths),
            'Медианная длина слов': st.median(word_lengths) if word_lengths else 0,
            'Средняя длина слов': st.mean(word_lengths) if word_lengths else 0.0,
            'Лексическое разнообразие': f"{lexical_div:.2f}",
            'Ошибки в тексте': num_errors,
            'Исправленный текст': corrected_text,
            'Ключевые слова': keyword_words[:10],
            'plot': self.plot_word_frequencies(text)
        }
        logger.info("Анализ текста завершен успешно")
        return result

processor = TextProcessor()

# Маршруты приложения
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form.get('text')
        file = request.files.get('file')

        if file:
            logger.info("Загрузка текста из файла")
            doc = Document(file)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif text:
            logger.info("Использование текста из формы")
        else:
            return jsonify({'error': 'Пожалуйста, введите текст или загрузите файл.'}), 400

        result = processor.analyze_text(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка при анализе текста: {str(e)}", exc_info=True)
        return jsonify({'error': 'Произошла ошибка при анализе текста.'}), 500

# Функция для выбора свободного порта 1024-65535
def get_available_port():
    while True:
        port = random.randint(1024, 65535)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', port))
            sock.close()
            return port
        except socket.error:
            continue

def run_app(app):
    port = get_available_port()
    print(f"Запускаем приложение на порте {port}")

    # Тунель NGROK
    NGROK_AUTH_TOKEN = os.getenv('NGROK_AUTH_TOKEN')
    if NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    else:
        logger.warning("NGROK_AUTH_TOKEN не установлен.")

    public_url = ngrok.connect(port)
    print(f"Общедоступный URL: {public_url}")

    app.run(host='0.0.0.0', port=port)

# Запуск приложения
if __name__ == "__main__":
    run_app(app)