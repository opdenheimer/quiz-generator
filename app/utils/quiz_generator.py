from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-e2e-qg")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-e2e-qg")

def generate_question(context, answer):
    input_text = f"answer: {answer}  context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_quiz(text, num_questions=5):
    sentences = sent_tokenize(text)
    quiz = []
    for sent in sentences[:num_questions * 2]:
        words = sent.split()
        if len(words) < 8:
            continue
        answer = words[-1].strip(".,")
        try:
            question = generate_question(sent, answer)
            options = [answer, "Option 1", "Option 2", "Option 3"]
            quiz.append({
                "question": question,
                "options": options,
                "answer": answer
            })
        except Exception:
            continue
        if len(quiz) >= num_questions:
            break
    return quiz
