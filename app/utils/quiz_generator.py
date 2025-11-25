import os
from pathlib import Path

# Set HuggingFace cache directory
cache_dir = Path.home() / '.cache' / 'huggingface' 
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir/ "transformers")
os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" 

import re
import random
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

model = None
tokenizer = None
model_loading = False


# -------------------------------------------------------
# B. LOAD MODEL (Lightweight & Cached)
# -------------------------------------------------------
def get_model():
    global model, tokenizer, model_loading
    
    if model is None:
        if model_loading:
            print("Model is already loading... waiting...")
            import time
            while model_loading:
                time.sleep(0.5)
            return model, tokenizer
        
        model_loading = True
        try:
            # Use smaller, faster model
            model_name = "t5-small"
            print(f"\n{'='*60}")
            print(f"Loading model: {model_name}")
            print(f"This may take 2-3 minutes on first load...")
            print(f"{'='*60}\n")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("✓ Tokenizer loaded")
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            print("✓ Model loaded successfully\n")
            model_loading = False
            
        except Exception as e:
            model_loading = False
            print(f"Error loading model: {e}")
            raise
    
    return model, tokenizer


# -------------------------------------------------------
# A. CLEAN TEXT
# -------------------------------------------------------
def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"rgba?\([^)]*\)", " ", text)
    text = re.sub(r"#([A-Fa-f0-9]{3,6})", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# -------------------------------------------------------
# C. EXTRACT ANSWER
# -------------------------------------------------------
def extract_answer(sentence):
    """Extract the most relevant word as answer from sentence"""
    try:
        words = [w.strip(".,!?()\"'") for w in sentence.split()]
        common_words = {'the','a','an','is','are','was','were','be','been','being',
                        'have','has','had','do','does','did','will','would','should',
                        'could','can','may','might','must','and','or','but','in','on',
                        'at','to','for','of','with','by','from','that','which'}
        words = [w for w in words if w.lower() not in common_words and len(w) > 2]

        if not words:
            return None

        # Prefer multi-word answers (named entities), else longest word
        multi_word_candidates = []
        toks = sentence.split()
        for i in range(len(toks)-1):
            pair = toks[i].strip(".,!?()\"'") + " " + toks[i+1].strip(".,!?()\"'")
            if pair[0].isupper():
                multi_word_candidates.append(pair)
        if multi_word_candidates:
            return multi_word_candidates[-1]

        proper_nouns = [w for w in words if w[0].isupper()]
        if proper_nouns:
            return proper_nouns[-1]

        longest = max(words, key=len)
        return longest
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None


# -------------------------------------------------------
# D. GENERATE QUESTION (Simplified & Faster)
# -------------------------------------------------------
def _template_question_from_context(context, answer):
    # Create a robust fill-in-the-blank or "which of the following" style question
    blanked = context.replace(answer, "_____")
    # If replacing didn't change text (case mismatch), try case-insensitive
    if blanked == context:
        import re
        blanked = re.sub(re.escape(answer), "_____", context, flags=re.IGNORECASE)
    # Keep it short
    if len(blanked) > 200:
        blanked = blanked[:200].rsplit(' ', 1)[0] + '...'
    return f"Fill in the blank: {blanked}"

def generate_question(context, answer):
    try:
        model_obj, tokenizer_obj = get_model()

        prompt = (
            f"Create a clear multiple-choice question (one sentence) based on the context below. "
            f"Make the correct answer exactly: {answer}\n\nContext: {context}\nQuestion:"
        )

        inputs = tokenizer_obj(prompt, return_tensors="pt", truncation=True, max_length=256)
        outputs = model_obj.generate(
            **inputs,
            max_length=80,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        q_text = tokenizer_obj.decode(outputs[0], skip_special_tokens=True).strip()

        # Post-checks: reject trivial T/F or single-word outputs
        short_or_tf = False
        if not q_text or len(q_text.split()) < 4:
            short_or_tf = True
        if q_text.lower() in {"true", "false", "yes", "no"}:
            short_or_tf = True
        if any(tok.lower() in q_text.lower() for tok in ["true/false", "true false", "yes/no"]):
            short_or_tf = True

        if short_or_tf:
            # fallback: build a reliable fill-in-the-blank question
            return _template_question_from_context(context, answer)

        # If the model produced something that accidentally includes the answer only (e.g., "True")
        # make sure it reads like a question
        if not q_text.endswith('?') and len(q_text.split()) <= 8:
            q_text = q_text.rstrip('.') + '?'

        return q_text
    except Exception as e:
        print(f"Error generating question: {e}")
        return _template_question_from_context(context, answer)


# -------------------------------------------------------
# E. CREATE DISTRACTORS
# -------------------------------------------------------
def create_distractors(correct_answer, topic_words):
    try:
        distractors = []
        seen = set([correct_answer.lower().strip()])

        # Prefer other topic words (cleaned)
        candidates = []
        for w in topic_words:
            wclean = w.strip(".,!?()\"'").strip()
            if not wclean:
                continue
            if wclean.lower() not in seen and len(wclean) > 2:
                candidates.append(wclean)
                seen.add(wclean.lower())

        # take up to 3 topic-word distractors
        for c in candidates:
            if len(distractors) >= 3:
                break
            # avoid duplicates and obvious numeric equality
            if c.lower() != correct_answer.lower():
                distractors.append(c)

        # If not enough, add plausible generic distractors with small variations
        generic = ["Not specified", "Unknown", "Cannot determine", "All of the above", "None of the above"]
        gi = 0
        while len(distractors) < 3 and gi < len(generic):
            g = generic[gi]
            if g.lower() not in seen:
                distractors.append(g)
                seen.add(g.lower())
            gi += 1

        # Final safety: ensure we have 3 distractors
        while len(distractors) < 3:
            distractors.append("Option " + chr(65 + len(distractors)))

        options = [correct_answer] + distractors[:3]
        random.shuffle(options)
        return options
    except Exception as e:
        print(f"Error creating distractors: {e}")
        return [correct_answer, "Option B", "Option C", "Option D"]


# -------------------------------------------------------
# F. MAIN QUIZ GENERATOR
# -------------------------------------------------------
def generate_quiz(text, num_questions=5):
    """Main function to generate quiz from text"""
    try:
        print("\n" + "="*60)
        print("STARTING QUIZ GENERATION")
        print("="*60)
        
        clean = clean_text(text)
        print(f"Text length: {len(clean)} characters")
        
        sentences = [s.strip() for s in sent_tokenize(clean) if len(s.split()) > 7]
        print(f"Found {len(sentences)} sentences")
        
        quiz = []
        
        for idx, sent in enumerate(sentences[:num_questions * 5]):
            print(f"\nQuestion {len(quiz) + 1}/{num_questions}: ", end="", flush=True)
            
            answer = extract_answer(sent)
            if not answer:
                print("✗ No answer")
                continue
            
            question = generate_question(sent, answer)
            if not question:
                print("✗ Generation failed")
                continue
            
            topic_words = [w.strip(".,!?()\"'") for w in sent.split() if len(w.strip(".,!?()\"'")) > 4]
            options = create_distractors(answer, topic_words)
            
            quiz.append({
                "question": question,
                "options": options,
                "answer": answer
            })
            print(f"✓ {question[:50]}...")
            
            if len(quiz) >= num_questions:
                break
        
        print(f"\n" + "="*60)
        print(f"Quiz generated: {len(quiz)} questions")
        print("="*60 + "\n")
        
        return quiz
    
    except Exception as e:
        print(f"ERROR in generate_quiz: {e}")
        import traceback
        traceback.print_exc()
        return []
