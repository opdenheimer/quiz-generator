from flask import Flask, render_template
import os

def create_app():
    app = Flask(__name__)
    
    # Config
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    
    # Register blueprints
    from app.quizgen.routes import quizgen_bp
    app.register_blueprint(quizgen_bp, url_prefix='/')
    
    return app
