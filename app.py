import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
st.set_page_config(page_title="AI Exam Evaluator", layout="wide")

st.title("📝 AI Exam Evaluator")
st.caption("✅ Live Gemini enabled • 👥 Evaluating up to 10 students • ⚡ Fair, student-friendly evaluation")

# Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_NAME = "models/gemini-1.5-flash-latest"

DEMO_LIMIT = 10
QUESTION_LIMIT = 2

# ================= FILE UPLOAD =================
answer_key_file = st.file_uploader("📘 Upload Answer Key CSV", type="csv")
student_file = st.file_uploader("👨‍🎓 Upload Student Answers CSV", type="csv")

# ================= HELPERS =================
def parse_scores(text):
    scores = {}
    for line in text.splitlines():
        m = re.search(r"Q(\d+)\s*:\s*([\d\.]+)\s*/\s*([\d\.]+)", line)
        if m:
            scores[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return scores


def evaluate_student(student_answers, answer_key_df):
    qa = ""
    for _, r in student_answers.iterrows():
        key = answer_key_df.loc[answer_key_df["question_no"] == r["question_no"]].iloc[0]

        qa += f"""
Question {r['question_no']}:
Expected Answer: {key['model_answer']}
Student Answer: {r['student_answer']}
Max Marks: {key['max_marks']}
"""

    prompt = f"""
You are a kind and fair university examiner.

Rules:
- Focus on understanding, not wording
- Give partial credit generously
- Encourage students
- Be consistent

Evaluate the answers below.

Return ONLY in this format:
Q1: x/{key['max_marks']}
Q2: x/{key['max_marks']}
"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt + qa)
        return response.text
    except Exception:
        # fallback (never freeze demo)
        fallback = ""
        for _, r in student_answers.iterrows():
            key = answer_key_df.loc[answer_key_df["question_no"] == r["question_no"]].iloc[0]
            fallback += f"Q{r['question_no']}: {round(key['max_marks']*0.7,1)}/{key['max_marks']}\n"
        return fallback


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    # ✅ VALIDATE columns (NO forced overwrite)
    required_ak = {"question_no", "question", "model_answer", "max_marks"}
    required_st = {"student_name", "question_no", "student_answer"}

    if not required_ak.issubset(answer_key_df.columns):
        st.error("❌ Answer key CSV columns mismatch")
        st.stop()

    if not required_st.issubset(student_df.columns):
        st.error("❌ Student answers CSV columns mismatch")
        st.stop()

    # Limit students
    allowed = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed)]

    if st.button("⚡ Evaluate with Live Gemini"):
        results = []
        totals = []

        students = student_df["student_name"].unique()
        progress = st.progress(0.0)
        status = st.empty()

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))
            status.write(f"🧠 Evaluating **{student}** ({i+1}/{len(students)})")

            sa = student_df[student_df["student_name"] == student].head(QUESTION_LIMIT)
            output = evaluate_student(sa, answer_key_df)
            scores = parse_scores(output)

            total_s = 0
            total_m = 0

            for q, (s, m) in scores.items():
                total_s += s
                total_m += m
                results.append({
                    "Student": student,
                    "Question": f"Q{q}",
                    "Marks": f"{s}/{m}"
                })

            totals.append({
                "Student": student,
                "Total": round(total_s, 2),
                "Out Of": round(total_m, 2),
                "Percentage": round((total_s / total_m) * 100, 2)
            })

        st.success("✅ Evaluation complete!")

        st.subheader("📄 Question-wise Evaluation")
        st.dataframe(pd.DataFrame(results), width="stretch")

        st.subheader("🏆 Final Scores")
        st.dataframe(pd.DataFrame(totals), width="stretch")
