from flask import Flask
from .text_processor import TextProcessor
from .utils import setup_logging, get_available_port

def create_app():
    app = Flask(__name__)
    
    setup_logging()
    
    app.text_processor = TextProcessor()
    
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app

__all__ = ['create_app', 'TextProcessor', 'setup_logging', 'get_available_port']