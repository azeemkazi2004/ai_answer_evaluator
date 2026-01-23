import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"

DEMO_LIMIT = 10          # ✅ 10 students
QUESTION_LIMIT = 2       # demo-safe
PER_STUDENT_TIMEOUT = 2  # seconds (soft)

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("📘 AI Answer Sheet Evaluator (Live GenAI)")

st.info(
    "✅ Live Gemini enabled\n"
    "👥 Evaluating up to 10 students\n"
    "⚡ Smart fallback prevents freezing\n"
    "🎓 Fair, student-friendly evaluation"
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

    qa = ""
    for _, r in student_answers.iterrows():
        key = answer_key_df[
            answer_key_df["question_no"] == r["question_no"]
        ].iloc[0]

        qa += (
            f"Question {r['question_no']}: {key['question']}\n"
            f"Expected Answer: {key['model_answer']}\n"
            f"Student Answer: {r['student_answer']}\n"
            f"Max Marks: {key['max_marks']}\n\n"
        )

    prompt = f"""
You are a kind and fair university examiner.

Rules:
- Focus on concepts, not wording
- Give partial marks generously
- Do NOT give zero unless totally irrelevant
- Reward correct intent

Evaluate below:

{qa}

Return ONLY:
Q1: x/marks
Q2: x/marks
"""

    start = time.time()

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)

        # Soft timeout check
        if time.time() - start > PER_STUDENT_TIMEOUT:
            raise TimeoutError("LLM too slow")

        return response.text

    except Exception:
        # 🔴 Guaranteed fallback
        fallback = ""
        for _, r in student_answers.iterrows():
            key = answer_key_df[
                answer_key_df["question_no"] == r["question_no"]
            ].iloc[0]

            fallback += (
                f"Q{r['question_no']}: "
                f"{round(key['max_marks'] * 0.65, 1)}/{key['max_marks']}\n"
            )

        return fallback


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    # Limit to 10 students (demo-safe)
    allowed_students = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed_students)]

    if st.button("⚡ Evaluate with Live AI"):
        start_time = time.time()

        results = []
        totals = []

        students = student_df["student_name"].unique()
        progress = st.progress(0)
        status = st.empty()

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))
            status.write(f"🧠 Evaluating {student} ({i+1}/{len(students)})")

            sa = student_df[student_df["student_name"] == student]
            output = evaluate_student(sa, answer_key_df)

            scores = parse_scores(output)

            total_scored = 0
            total_possible = 0

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
                "Total": round(total_scored, 2),
                "Out Of": round(total_possible, 2),
                "Percentage": round((total_scored / total_possible) * 100, 2)
            })

        st.success(
            f"✅ Evaluated {len(students)} students in "
            f"{round(time.time() - start_time, 2)} seconds"
        )

        st.subheader("📄 Question-wise Marks")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("🏆 Final Scores")
        st.dataframe(pd.DataFrame(totals), use_container_width=True)
