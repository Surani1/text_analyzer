import os
import logging
import logging.config
import yaml
import random
import socket

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Загрузка конфигурации логирования
    with open('config/logging.yaml', 'r', encoding='utf-8') as f:
        logging_config = yaml.safe_load(f)
    
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.debug("Программа запущена успешно. Начало сбора логов")

def get_available_port():
    while True:
        port = random.randint(1024, 65535)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue