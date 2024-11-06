# Используем официальный легковесный образ Python в качестве базового
FROM python:3.10-slim

# Устанавливаем переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Обновляем pip до последней версии
RUN pip install --upgrade pip

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем и устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта в контейнер
COPY . .

# Загружаем необходимые данные NLTK
RUN python -m nltk.downloader punkt stopwords

# Указываем порт, который будет использоваться приложением
EXPOSE 5000

# Определяем команду запуска приложения
CMD ["python", "app.py"]