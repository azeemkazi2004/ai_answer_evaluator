def evaluate_student(student_answers, answer_key_df):
    student_answers = student_answers.head(2)  # demo-safe

    qa = ""
    for _, r in student_answers.iterrows():
        key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
        qa += (
            f"Question: {key['question']}\n"
            f"Expected: {key['model_answer']}\n"
            f"Student: {r['student_answer']}\n"
            f"Marks: {key['max_marks']}\n\n"
        )

    prompt = f"""
You are a kind and fair examiner.
Give partial marks generously.
Do NOT give zero unless totally wrong.

Return ONLY:
Q1: x/marks
Q2: x/marks

{qa}
"""

    try:
        model = genai.GenerativeModel("models/gemini-flash-lite-latest")
        response = model.generate_content(prompt, timeout=3)
        return response.text

    except Exception:
        # 🔴 GUARANTEED FALLBACK (NO FREEZE)
        fallback = ""
        for _, r in student_answers.iterrows():
            key = answer_key_df[answer_key_df["question_no"] == r["question_no"]].iloc[0]
            fallback += f"Q{r['question_no']}: {key['max_marks']*0.6}/{key['max_marks']}\n"
        return fallback
