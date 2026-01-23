import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"

DEMO_LIMIT = 3        # live demo safety
QUESTION_LIMIT = 3    # enough to show reasoning

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("📘 AI Answer Sheet Evaluator (Live GenAI)")

st.info(
    "Live Gemini evaluation enabled.\n"
    "Demo uses limited students/questions for reliability."
)

# ================= FILE UPLOAD =================
answer_key_file = st.file_uploader("Upload Answer Key CSV", type="csv")
student_file = st.file_uploader("Upload Student Answers CSV", type="csv")

# ================= HELPERS =================
def parse_scores(text):
    scores = {}
    for line in text.splitlines():
        m = re.search(r"Q(\d+)\s*:\s*([\d\.]+)\s*/\s*([\d\.]+)", line)
        if m:
            scores[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return scores


def evaluate_student(student_answers, answer_key_df):
    student_answers = student_answers.head(QUESTION_LIMIT)

    qa_block = ""
    for _, row in student_answers.iterrows():
        key = answer_key_df[
            answer_key_df["question_no"] == row["question_no"]
        ].iloc[0]

        qa_block += (
            f"Question {row['question_no']}: {key['question']}\n"
            f"Expected Answer: {key['model_answer']}\n"
            f"Student Answer: {row['student_answer']}\n"
            f"Maximum Marks: {key['max_marks']}\n\n"
        )

    prompt = f"""
You are a kind and fair university examiner.

Evaluation rules:
- Focus on understanding and key concepts, not exact wording
- Give partial marks generously if the idea is present
- Do NOT give zero unless the answer is completely irrelevant
- Reward correct intent even if explanation is short
- Marks should feel realistic and encouraging

Evaluate the following answers:

{qa_block}

Return ONLY marks in this format:
Q1: x/marks
Q2: x/marks
"""

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    # Demo safety cut
    allowed_students = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed_students)]

    if st.button("⚡ Evaluate with Live AI"):
        start = time.time()
        results, totals = [], []

        students = student_df["student_name"].unique()
        progress = st.progress(0)

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))
            st.write(f"Evaluating {student}...")

            sa = student_df[student_df["student_name"] == student]
            output = evaluate_student(sa, answer_key_df)
            scores = parse_scores(output)

            total_scored, total_possible = 0, 0

            for q, (s, m) in scores.items():
                total_scored += s
                total_possible += m
                results.append({
                    "Student": student,
                    "Question": q,
                    "Marks": f"{s}/{m}"
                })

            totals.append({
                "Student": student,
                "Total": total_scored,
                "Out Of": total_possible,
                "Percentage": round((total_scored / total_possible) * 100, 2)
            })

        st.success(f"✅ Completed in {round(time.time() - start, 2)} seconds")

        st.subheader("📄 Question-wise Marks")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("🏆 Final Scores")
        st.dataframe(pd.DataFrame(totals), use_container_width=True)
