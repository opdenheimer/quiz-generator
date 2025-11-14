from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
from app.utils.pdf_utils import extract_text_from_pdf
from app.utils.quiz_generator import generate_quiz

quizgen_bp = Blueprint('quizgen', __name__, template_folder='../../templates')

QUIZ_CACHE = {}

@quizgen_bp.route('/upload', methods=['GET', 'POST'])
def upload_doc():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return "No file uploaded", 400
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        questions = generate_quiz(text, num_questions=5)

        quiz_id = len(QUIZ_CACHE) + 1
        QUIZ_CACHE[quiz_id] = questions
        return redirect(url_for('quizgen.play_quiz', quiz_id=quiz_id))
    return render_template('upload.html')

@quizgen_bp.route('/quiz/<int:quiz_id>')
def play_quiz(quiz_id):
    questions = QUIZ_CACHE.get(quiz_id)
    if not questions:
        return "Quiz not found", 404
    return render_template('quiz.html', questions=questions, quiz_id=quiz_id)

@quizgen_bp.route('/submit/<int:quiz_id>', methods=['POST'])
def submit_quiz(quiz_id):
    questions = QUIZ_CACHE.get(quiz_id)
    score = 0
    for i, q in enumerate(questions):
        ans = request.form.get(f'q{i}')
        if ans and ans.lower().strip() == q['answer'].lower():
            score += 1
    return render_template('result.html', score=score, total=len(questions))
