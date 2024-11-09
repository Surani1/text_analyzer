from flask import Blueprint, render_template, request, jsonify, current_app
import logging
from docx import Document
from io import BytesIO

main = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form.get('text')
        file = request.files.get('file')

        if file:
            logger.info("Загрузка текста из файла")
            doc = Document(BytesIO(file.read()))
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif text:
            logger.info("Использование текста из формы")
        else:
            return jsonify({'error': 'Пожалуйста, введите текст или загрузите файл.'}), 400

        # Используем TextProcessor, который теперь является атрибутом приложения
        result = current_app.text_processor.analyze_text(text)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Ошибка при анализе текста: {str(e)}", exc_info=True)
        return jsonify({'error': 'Произошла ошибка при анализе текста.'}), 500