"""
app.py — Gradio web interface for the Unofficial Berkeley CS Guide.

Run with: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from query import ask


def handle_query(question: str):
    """
    Called when the user submits a question.
    Returns (answer_text, sources_text, retrieved_context_text).
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    result = ask(question)

    answer = result["answer"]

    sources_text = "\n".join(f"• {s}" for s in result["sources"])

    context_lines = []
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        context_lines.append(
            f"[{i}] {chunk['source']} (similarity distance: {chunk['distance']:.4f})\n"
            f"{chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}"
        )
    retrieved_context = "\n\n".join(context_lines)

    return answer, sources_text, retrieved_context


EXAMPLE_QUESTIONS = [
    "What is the difference between taking CS61B with Hilfinger vs. Hug?",
    "What topics should I study for the CS186 exam?",
    "Which dining hall has the shortest wait times on campus?",
    "When should I start applying for software engineering internships at Amazon?",
    "How long is the wait to see a counselor at CAPS for mental health?",
    "What are the Pintos projects in CS162?",
    "What GPA do I need to declare the CS major at Berkeley?",
    "What neighborhoods are good for off-campus housing near Berkeley?",
    "What is the Scheme project in CS61A and how hard is it?",
    "How does the CS170 homework policy work with group collaboration?",
]

with gr.Blocks(title="The Unofficial Berkeley CS Guide") as demo:
    gr.Markdown(
        """
        # 🐻 The Unofficial Berkeley CS Guide
        ### A RAG system over student-generated knowledge about UC Berkeley CS courses, campus life, housing, and more.

        Ask any question about professors, courses, exams, dining, housing, internships, or mental health resources.
        Answers are grounded in real student reviews and Reddit posts and other sources (not offical guide from UC Berkeley).
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What is the difference between Hilfinger and Hug for CS61B?",
                lines=3,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_input,
                label="Example Questions",
            )

        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                label="Answer",
                lines=10,
                interactive=False,
            )
            sources_output = gr.Textbox(
                label="Retrieved From (Source Documents)",
                lines=4,
                interactive=False,
            )

    with gr.Accordion("🔍 Retrieved Context Chunks (for transparency)", open=False):
        context_output = gr.Textbox(
            label="Top 5 Retrieved Chunks",
            lines=15,
            interactive=False,
        )

    gr.Markdown(
        """
        ---
        **About this system:** Answers are generated using `llama-3.3-70b-versatile` via Groq,
        grounded in 14 documents from Rate My Professors, Reddit r/berkeley, Yelp, and student blogs.
        Embeddings: `all-MiniLM-L6-v2`. Vector store: ChromaDB (local).
        """
    )

    submit_btn.click(
        handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, context_output],
    )
    question_input.submit(
        handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output, context_output],
    )
    clear_btn.click(
        lambda: ("", "", ""),
        outputs=[answer_output, sources_output, context_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="blue"))
