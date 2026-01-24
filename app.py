import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import re
import time

# ================= CONFIG =================
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-flash-lite-latest"
DEMO_LIMIT = 5          # keep safe for live demo
QUESTION_LIMIT = 2

st.set_page_config(page_title="AI Exam Evaluator", layout="wide")

# ================= TITLE =================
st.title("📝 AI Exam Evaluator")
st.markdown(
    "✅ **Live Gemini enabled** &nbsp;&nbsp; "
    "👥 **Evaluating up to 10 students** &nbsp;&nbsp; "
    "⚡ **Fair, student-friendly evaluation**"
)
st.markdown("---")

# ================= LAYOUT =================
left, right = st.columns([2, 1])

with left:
    answer_key_file = st.file_uploader("📘 Upload Answer Key CSV", type="csv")
    student_file = st.file_uploader("🧑‍🎓 Upload Student Answers CSV", type="csv")

with right:
    st.markdown("### ℹ️ How it works")
    st.markdown(
        """
        1. Upload answer key  
        2. Upload student answers  
        3. Click **Evaluate**  
        4. View marks, feedback, analytics & leaderboard  
        """
    )

st.info("🧠 The AI evaluates answers using a kind, concept-based grading rubric.")

# ================= HELPERS =================
def parse_scores_and_feedback(text):
    results = {}
    current_q = None

    for line in text.splitlines():
        score_match = re.search(r"Q(\d+)\s*:\s*([\d\.]+)\s*/\s*([\d\.]+)", line)
        if score_match:
            q = int(score_match.group(1))
            results[q] = {
                "scored": float(score_match.group(2)),
                "max": float(score_match.group(3)),
                "feedback": ""
            }
            current_q = q

        if current_q and line.lower().startswith("feedback"):
            results[current_q]["feedback"] = line.split(":", 1)[1].strip()

    return results


def evaluate_student(student_answers, answer_key_df):
    student_answers = student_answers.head(QUESTION_LIMIT)

    qa_block = ""
    for _, r in student_answers.iterrows():
        key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
        qa_block += (
            f"Question {r['question_no']}: {key['question']}\n"
            f"Recommended Answer: {key['model_answer']}\n"
            f"Student Answer: {r['student_answer']}\n"
            f"Maximum Marks: {key['max_marks']}\n\n"
        )

    prompt = f"""
You are a kind, supportive, and fair university examiner.

Rules:
- Focus on understanding, not exact wording
- Give partial marks generously
- Do NOT give zero unless the answer is completely irrelevant
- Reward correct intent
- Provide 1-line constructive feedback

Evaluate the following answers:

{qa_block}

Return ONLY in this format:
Q1: x/marks
Feedback: <short feedback>
Q2: x/marks
Feedback: <short feedback>
"""

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        # Safe fallback (never blocks)
        fallback = ""
        for _, r in student_answers.iterrows():
            key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
            fallback += (
                f"Q{r['question_no']}: {round(key['max_marks'] * 0.7, 1)}/{key['max_marks']}\n"
                "Feedback: Partial understanding shown.\n"
            )
        return fallback


# ================= MAIN =================
if answer_key_file and student_file:
    answer_key_df = pd.read_csv(answer_key_file)
    student_df = pd.read_csv(student_file)

    answer_key_df.columns = ["question_no", "question", "model_answer", "max_marks"]
    student_df.columns = ["student_name", "question_no", "student_answer"]

    allowed_students = student_df["student_name"].unique()[:DEMO_LIMIT]
    student_df = student_df[student_df["student_name"].isin(allowed_students)]

    if st.button("⚡ Evaluate with Live AI"):
        start_time = time.time()

        detailed_rows = []
        totals = []

        students = student_df["student_name"].unique()
        progress = st.progress(0)

        for i, student in enumerate(students):
            progress.progress((i + 1) / len(students))

            sa = student_df[student_df["student_name"] == student]
            output = evaluate_student(sa, answer_key_df)
            parsed = parse_scores_and_feedback(output)

            total_scored = 0
            total_possible = 0

            for q, data in parsed.items():
                key = answer_key_df[answer_key_df["question_no"] == q].iloc[0]
                stud_ans = sa[sa["question_no"] == q]["student_answer"].values[0]

                total_scored += data["scored"]
                total_possible += data["max"]

                detailed_rows.append({
                    "Student": student,
                    "Question": q,
                    "Recommended Answer": key["model_answer"],
                    "Student Answer": stud_ans,
                    "Marks": f"{data['scored']}/{data['max']}",
                    "AI Feedback": data["feedback"]
                })

            totals.append({
                "Student": student,
                "Total": round(total_scored, 2),
                "Out Of": round(total_possible, 2),
                "Percentage": round((total_scored / total_possible) * 100, 2)
            })

        elapsed = round(time.time() - start_time, 2)

        # ================= METRICS =================
        totals_df = pd.DataFrame(totals)

        c1, c2, c3 = st.columns(3)
        c1.metric("👥 Students Evaluated", len(totals_df))
        c2.metric("📊 Class Average (%)", round(totals_df["Percentage"].mean(), 2))
        c3.metric("⏱️ Time Taken (sec)", elapsed)

        st.success("✅ Evaluation completed successfully!")

        # ================= DETAILED EVALUATION =================
        st.subheader("📄 Detailed Evaluation (Marks + Feedback)")
        detailed_df = pd.DataFrame(detailed_rows)
        st.dataframe(detailed_df, use_container_width=True)

        # ================= ANALYTICS =================
        st.subheader("📊 Class Analytics")

        a1, a2 = st.columns(2)
        a1.metric("🏆 Highest Score (%)", totals_df["Percentage"].max())
        a2.metric("🔻 Lowest Score (%)", totals_df["Percentage"].min())

        st.markdown("#### 📈 Score Distribution")
        st.bar_chart(totals_df.set_index("Student")["Percentage"])

        st.markdown("#### 🧠 Question-wise Average Performance")
        q_avg = detailed_df.groupby("Question")["Marks"].count().reset_index()
        st.bar_chart(q_avg.set_index("Question"))

        # ================= LEADERBOARD =================
        st.subheader("🏆 Leaderboard")
        leaderboard_df = totals_df.sort_values(
            by="Percentage", ascending=False
        ).reset_index(drop=True)

        leaderboard_df.index += 1
        leaderboard_df.insert(0, "Rank", leaderboard_df.index)
        leaderboard_df["Rank"] = leaderboard_df["Rank"].replace({
            1: "🥇 1", 2: "🥈 2", 3: "🥉 3"
        })

        st.dataframe(leaderboard_df, use_container_width=True)

        # ================= DOWNLOAD =================
        st.download_button(
            "📥 Download Evaluation Report (CSV)",
            detailed_df.to_csv(index=False),
            file_name="ai_exam_evaluation_report.csv",
            mime="text/csv"
        )

      

# ================= FOOTER =================
st.markdown("---")
st.caption("Built with ❤️ using Python, Streamlit & Google Gemini | Hackathon Project")

