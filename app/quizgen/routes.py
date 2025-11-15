from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
from app.utils.pdf_utils import extract_text_from_pdf
from app.utils.quiz_generator import generate_quiz

quizgen_bp = Blueprint('quizgen', __name__, template_folder='../../templates')

QUIZ_CACHE = {}

@quizgen_bp.route('/')
@quizgen_bp.route('/home')
def home():
    return render_template('upload.html')  # Changed to upload.html

@quizgen_bp.route('/upload', methods=['GET', 'POST'])
def upload_doc():
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            
            if not file or file.filename == '':
                return render_template('upload.html', error="Please select a file.")
            
            if not file.filename.lower().endswith('.pdf'):
                return render_template('upload.html', error="Only PDF files allowed.")
            
            upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)
            
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            print(f"File saved to: {file_path}")

            text = extract_text_from_pdf(file_path)
            
            if not text or len(text.strip()) < 50:
                return render_template('upload.html', error="PDF has too little text.")
            
            print("Generating quiz... (this may take 1-2 minutes on first run)")
            questions = generate_quiz(text, num_questions=5)
            
            if not questions:
                return render_template('upload.html', error="Could not generate quiz. Try another PDF.")

            quiz_id = len(QUIZ_CACHE) + 1
            QUIZ_CACHE[quiz_id] = questions
            print(f"✓ Quiz {quiz_id} ready!")
            
            return redirect(url_for('quizgen.play_quiz', quiz_id=quiz_id))
        
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template('upload.html', error=str(e))
    
    return render_template('upload.html')

@quizgen_bp.route('/quiz/<int:quiz_id>', methods=['GET'])
def play_quiz(quiz_id):
    questions = QUIZ_CACHE.get(quiz_id)
    if not questions:
        return render_template('upload.html', error=f"Quiz {quiz_id} not found."), 404
    return render_template('quiz.html', questions=questions, quiz_id=quiz_id)

@quizgen_bp.route('/submit/<int:quiz_id>', methods=['POST'])
def submit_quiz(quiz_id):
    try:
        questions = QUIZ_CACHE.get(quiz_id)
        if not questions:
            return render_template('upload.html', error="Quiz not found"), 404
        
        score = 0
        total = len(questions)
        user_answers={}
        
        for i, q in enumerate(questions):
            user_answer = request.form.get(f'q{i}')
            user_answers[f'q{i}'] = user_answer
            correct_answer = q['answer']
            
            if user_answer and user_answer.lower().strip() == correct_answer.lower().strip():
                score += 1
        
        percentage = (score/total)*100 if total > 0 else 0
        return render_template('result.html', score=score, total=total, quiz_id=quiz_id,
                               percentage=round(percentage,2),questions=questions,user_answers=user_answers)
    
    except Exception as e:
        print(f"Error submitting quiz: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('upload.html', error=f"Error: {str(e)}"), 500
