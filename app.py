import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"  # fastest & stable
DEMO_LIMIT = 15  # demo-safe limit (set None for full run)

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("⚡ AI Answer Sheet Evaluator (Fast & Stable Demo)")

st.info(
    "Demo mode evaluates first 15 students for speed.\n"
    "Uses batched evaluation (1 AI call per student)."
)

# ================= FILE UPLOAD =================
answer_key_file = st.file_uploader("Upload Answer Key CSV", type="csv")
student_file = st.file_uploader("Upload Student Answers CSV", type="csv")

# ================= HELPERS =================
def parse_scores(text):
    """
    Extracts:
    Q1: 2.5/5
    Q2: 3/5
    """
    scores = {}
    for line in text.splitlines():
        match = re.search(r"Q(\d+)\s*:\s*([\d\.]+)\s*/\s*([\d\.]+)", line)
        if match:
            q = int(match.group(1))
            scored = float(match.group(2))
            total = float(match.group(3))
            scores[q] = (scored, total)
    return scores


@st.cache_data(show_spinner=False)
def evaluate_student(student_name, student_answers, answer_key_df):
    """
    ONE Gemini call per student (cached).
    No signals — Streamlit safe.
    """
    qa_block = ""
    for _, row in student_answers.iterrows():
        key = answer_key_df[answer_key_df["question_no"] == row["question_no"]].iloc[0]
        qa_block += f"""
Q{row['question_no']}. {key['question']}
Model Answer: {key['model_answer']}
Student Answer: {row['student_answer']}
Max Marks: {key['max_marks']}
"""

    prompt = f"""
You are a strict but fair exam evaluator.

Evaluate answers for student: {student_name}

{qa_block}

Rules:
- Be consistent
- Allow partial marks
- Use given max marks

Respond ONLY in this format:
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

    # Force schema (robust against CSV header issues)
    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    if st.button("⚡ Evaluate (Fast Mode)"):
        results = []
        totals = []

        students = student_df["student_name"].unique()

        # 🔴 DEMO LIMIT
        if DEMO_LIMIT:
            students = students[:DEMO_LIMIT]

        progress = st.progress(0)
        status = st.empty()

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))
            status.write(f"Evaluating {student} ({i+1}/{len(students)})")

            student_answers = student_df[student_df["student_name"] == student]
            ai_output = evaluate_student(student, student_answers, answer_key_df)

            score_map = parse_scores(ai_output)

            total_scored = 0
            total_possible = 0

            for q_no, (scored, maxm) in score_map.items():
                total_scored += scored
                total_possible += maxm

                results.append({
                    "Student": student,
                    "Question": q_no,
                    "Marks": f"{scored}/{maxm}"
                })

            totals.append({
                "Student": student,
                "Total Scored": total_scored,
                "Total Possible": total_possible,
                "Percentage": round((total_scored / total_possible) * 100, 2)
            })

        st.success("✅ Evaluation completed")

        st.subheader("📄 Per-Question Marks")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("🏆 Final Totals")
        st.dataframe(pd.DataFrame(totals), use_container_width=True)
