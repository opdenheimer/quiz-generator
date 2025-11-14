from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Register the quizgen blueprint
    from app.quizgen.routes import quizgen_bp
    app.register_blueprint(quizgen_bp, url_prefix='/quizgen')

    # Simple home route
    @app.route('/')
    def home():
        return '''
        <h2>AI Quiz Generator 🧠</h2>
        <p><a href="/quizgen/upload">Upload a Document to Generate Quiz</a></p>
        '''

    return app
