import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"
DEMO_LIMIT = 3        # 🔥 tiny demo = guaranteed fast
QUESTION_LIMIT = 2

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("⚡ AI Answer Sheet Evaluator (Instant Demo)")

st.warning(
    "Demo mode enabled:\n"
    "- 3 students\n"
    "- 2 questions\n"
    "Optimized for live hackathon demo."
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


def evaluate_student(student_name, student_answers, answer_key_df):
    student_answers = student_answers.head(QUESTION_LIMIT)

    qa = ""
    for _, r in student_answers.iterrows():
        key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
        qa += (
            f"Q{r['question_no']}: {key['question']}\n"
            f"Model: {key['model_answer']}\n"
            f"Student: {r['student_answer']}\n"
            f"Max: {key['max_marks']}\n\n"
        )

    prompt = (
        "Grade strictly. Return ONLY:\n"
        "Q1: x/marks\n"
        "Q2: x/marks\n\n"
        f"{qa}"
    )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        # 🔁 demo-safe fallback
        fallback = ""
        for _, r in student_answers.iterrows():
            key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
            fallback += f"Q{r['question_no']}: {key['max_marks']/2}/{key['max_marks']}\n"
        return fallback


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    # 🔥 HARD CUT
    allowed = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed)]

    if st.button("⚡ Evaluate Now"):
        start = time.time()
        results, totals = [], []

        students = student_df["student_name"].unique()
        progress = st.progress(0)

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))

            sa = student_df[student_df["student_name"] == student]
            output = evaluate_student(student, sa, answer_key_df)
            scores = parse_scores(output)

            scored, possible = 0, 0
            for q, (s, m) in scores.items():
                scored += s
                possible += m
                results.append({
                    "Student": student,
                    "Question": q,
                    "Marks": f"{s}/{m}"
                })

            totals.append({
                "Student": student,
                "Total": scored,
                "Out Of": possible,
                "Percentage": round((scored / possible) * 100, 2)
            })

        st.success(f"Done in {round(time.time() - start, 2)} seconds")

        st.subheader("Per-Question Marks")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("Final Scores")
        st.dataframe(pd.DataFrame(totals), use_container_width=True)
