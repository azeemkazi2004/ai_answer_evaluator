import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="AI Answer Sheet Evaluator", layout="wide")
st.title("📘 Answer Sheet Evaluator (Demo Mode)")

st.info(
    "Demo Mode enabled (no live LLM calls).\n"
    "AI evaluation runs asynchronously in production."
)

answer_key_file = st.file_uploader("Upload Answer Key CSV", type="csv")
student_file = st.file_uploader("Upload Student Answers CSV", type="csv")

def score_answer(model_answer, student_answer, max_marks):
    model_keywords = set(re.findall(r"\w+", model_answer.lower()))
    student_keywords = set(re.findall(r"\w+", student_answer.lower()))

    if not model_keywords:
        return 0

    overlap = model_keywords.intersection(student_keywords)
    score = (len(overlap) / len(model_keywords)) * max_marks
    return round(score, 2)

if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    if st.button("Evaluate (Instant Demo)"):
        results = []

        for _, row in student_df.iterrows():
            key = answer_key_df[
                answer_key_df["question_no"] == row["question_no"]
            ].iloc[0]

            marks = score_answer(
                key["model_answer"],
                row["student_answer"],
                key["max_marks"]
            )

            results.append({
                "Student": row["student_name"],
                "Question": row["question_no"],
                "Marks": f"{marks}/{key['max_marks']}"
            })

        results_df = pd.DataFrame(results)
        st.subheader("📄 Per-Question Marks")
        st.dataframe(results_df)

        totals = (
            results_df
            .assign(
                scored=lambda x: x["Marks"].str.split("/").str[0].astype(float),
                total=lambda x: x["Marks"].str.split("/").str[1].astype(float)
            )
            .groupby("Student", as_index=False)
            .agg(
                Total_Scored=("scored", "sum"),
                Total_Possible=("total", "sum")
            )
        )

        totals["Percentage"] = (
            totals["Total_Scored"] / totals["Total_Possible"] * 100
        ).round(2)

        st.subheader("🏆 Final Scores")
        st.dataframe(totals)
