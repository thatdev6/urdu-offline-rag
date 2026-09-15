import os
import sys

import gradio as gr

import gradio.networking as _gr_networking
_gr_networking.url_ok = lambda url: True

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))
from rag import RAGPipeline  # noqa: E402

print("Initializing RAG pipeline (loading embedding model + FAISS index)...")
rag = RAGPipeline()
print("Pipeline ready.")

LIMITATIONS_NOTE = (
    "**Known limitations** (see `LIMITATIONS.md` and `FINDINGS.md` in the repo for full detail): "
    "casual Roman Urdu phrasing sometimes retrieves the wrong chunk even when the answer exists "
    "in the corpus, since the source documents are formally worded in English. Two spelling/"
    "translation-based fixes were tested and neither reliably closed that gap — a reranking or "
    "hybrid search step is the more promising next step, not yet built."
)


def answer_query(query: str):
    if not query or not query.strip():
        return "Please enter a question.", "", ""

    result = rag.answer(query, verbose=False)

    sources_md = ""
    for c in result["retrieved_chunks"]:
        snippet = c["text"][:200].replace("\n", " ")
        sources_md += f"**[{c['score']:.3f}] {c['source']}**\n\n{snippet}...\n\n---\n\n"

    grounded = "Grounded in retrieved documents" if result["sources"] else "No matching source found"

    return result["answer"], sources_md, grounded


with gr.Blocks(title="Urdu Offline RAG") as demo:
    gr.Markdown(
        "# Urdu Offline RAG\n"
        "Ask a question about university policy (fees, exams, hostel, library, discipline, etc.) "
        "in **Roman Urdu, Urdu script, or English**. Answers are grounded in real university "
        "regulation documents — retrieval + generation both run on a small quantized model, "
        "entirely CPU, no external API calls at inference time."
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Hostel fine kitni hai agar late aaon?",
                lines=2,
            )
            submit_btn = gr.Button("Ask", variant="primary")
            answer_output = gr.Textbox(label="Answer", lines=4, interactive=False)
            grounding_output = gr.Markdown()

        with gr.Column(scale=1):
            gr.Markdown("### Retrieved chunks")
            sources_output = gr.Markdown()

    gr.Markdown(LIMITATIONS_NOTE)

    submit_btn.click(
        fn=answer_query,
        inputs=query_input,
        outputs=[answer_output, sources_output, grounding_output],
    )
    query_input.submit(
        fn=answer_query,
        inputs=query_input,
        outputs=[answer_output, sources_output, grounding_output],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
