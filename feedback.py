# feedback.py
import json
def collect_feedback(question, answer, rating):
    record = {
        "question": question,
        "answer": answer,
        "rating": rating
    }
    with open("feedback_store.json", "a") as f:
        f.write(json.dumps(record) + "\n")