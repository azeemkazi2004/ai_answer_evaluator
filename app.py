import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"  # fastest & safest
DEMO_LIMIT = 5        # 🔴 keep small for guaranteed demo
QUESTION_LIMIT = 3    # 🔴 limit questions per student

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("⚡ AI Answer Sheet Evaluator (Hackathon Demo)")

st.info(
    "Demo mode enabled:\n"
    "- First 5 students\n"
    "- First 3 questions per student\n"
    "This avoids LLM timeouts on Streamlit Cloud."
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
def warm_up_model():
    """Warm up Gemini to avoid cold start freeze"""
    model = genai.GenerativeModel(MODEL_NAME)
    model.generate_content("Reply OK")


@st.cache_data(show_spinner=False)
def evaluate_student(student_name, student_answers, answer_key_df):
    """
    ONE Gemini call per student
    Small prompt + Gemini timeout to avoid 504
    """
    student_answers = student_answers.head(QUESTION_LIMIT)

    qa_block = ""
    for _, row in student_answers.iterrows():
        key = answer_key_df[
            answer_key_df["question_no"] == row["question_no"]
        ].iloc[0]

        qa_block += (
            f"Q{row['question_no']}: {key['question']}\n"
            f"Model: {key['model_answer']}\n"
            f"Student: {row['student_answer']}\n"
            f"Max: {key['max_marks']}\n\n"
        )

    prompt = (
        "Grade strictly and fairly.\n"
        "Return ONLY in this format:\n"
        "Q1: x/marks\n"
        "Q2: x/marks\n\n"
        f"{qa_block}"
    )

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        request_options={"timeout": 15}  # Gemini-side timeout
    )

    return response.text


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    # Force schema (robust)
    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    # 🔴 HARD CUT DATA BEFORE ANY AI CALL
    allowed_students = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed_students)]

    if st.button("⚡ Evaluate (Demo Mode)"):
        start_time = time.time()

        st.info("Warming up AI model…")
        warm_up_model()

        results = []
        totals = []

        students = student_df["student_name"].unique()
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

        elapsed = round(time.time() - start_time, 2)

        st.success(f"✅ Completed in {elapsed} seconds")

        st.subheader("📄 Per-Question Marks")
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("🏆 Final Totals")
        st.dataframe(pd.DataFrame(totals), use_container_width=True)
