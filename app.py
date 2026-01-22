import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re

# ---- CONFIGURE GEMINI (STREAMLIT WAY) ----
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("📘 AI Answer Sheet Evaluator")

st.write("Upload the answer key and student answers (CSV files)")

# ---- FILE UPLOADS ----
answer_key_file = st.file_uploader("Upload Answer Key CSV", type="csv")
student_file = st.file_uploader("Upload Student Answers CSV", type="csv")

# ---- AI EVALUATION FUNCTION ----
def evaluate_single_answer(question, model_answer, student_answer, max_marks):
    prompt = f"""
You are a strict but fair exam evaluator.

Question:
{question}

Model Answer:
{model_answer}

Student Answer:
{student_answer}

Respond EXACTLY in this format:

Score: X/{max_marks}
Correct Points:
- ...
Missing Points:
- ...
"""
    model = genai.GenerativeModel("models/gemini-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# ---- SCORE EXTRACTION ----
def extract_score(text):
    match = re.search(r"Score:\s*([\d\.]+)\s*/\s*([\d\.]+)", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

# ---- MAIN LOGIC ----
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    # Force column names (robust)
    student_df.columns = ["student_name", "question_no", "student_answer"]
    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]

    if st.button("Evaluate Answers"):
        results = []

        for _, row in student_df.iterrows():
            key = answer_key_df[
                answer_key_df["question_no"] == row["question_no"]
            ].iloc[0]

            evaluation = evaluate_single_answer(
                key["question"],
                key["model_answer"],
                row["student_answer"],
                key["max_marks"]
            )

            scored, total = extract_score(evaluation)

            results.append({
                "Student": row["student_name"],
                "Question": row["question_no"],
                "Marks": f"{scored}/{total}",
                "Evaluation": evaluation
            })

        results_df = pd.DataFrame(results)
        st.subheader("📄 Per-Question Evaluation")
        st.dataframe(results_df)

        # ---- TOTALS ----
        totals_df = (
            results_df
            .assign(
                marks_scored=results_df["Evaluation"].apply(lambda x: extract_score(x)[0]),
                max_marks=results_df["Evaluation"].apply(lambda x: extract_score(x)[1])
            )
            .groupby("Student", as_index=False)
            .agg(
                Total_Scored=("marks_scored", "sum"),
                Total_Possible=("max_marks", "sum")
            )
        )

        totals_df["Percentage"] = (
            totals_df["Total_Scored"] / totals_df["Total_Possible"] * 100
        ).round(2)

        st.subheader("🏆 Final Scores")
        st.dataframe(totals_df)
