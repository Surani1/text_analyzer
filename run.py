import os
from app.__init__ import create_app
from app.utils import get_available_port
from pyngrok import ngrok
from app.main import logger

if __name__ == "__main__":
    app = create_app()
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