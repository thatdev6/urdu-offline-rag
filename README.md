# Urdu Offline RAG

An offline, bilingual retrieval augmented generation system built against real public documents from a Pakistani university. It runs entirely on CPU with no internet access at inference time, and answers questions in whichever form they're asked in: Roman Urdu, Urdu script, or English.

This is a personal reference implementation, not a Manar clone. I built it to put real numbers behind claims I would otherwise just be asserting in an interview: how well does a small quantized model handle Urdu, how much does retrieval quality actually matter, and where does casual Roman Urdu break things that formal English doesn't.

## What it does

Given a question about university policy (fees, exams, hostel rules, library rules, discipline, and so on), the system:

1. Embeds the query using BGE-M3, a multilingual embedding model
2. Retrieves the most relevant chunks from a FAISS index built over the university's public regulation documents
3. Sends the retrieved chunks and the question to a local, quantized Qwen3 model running through llama.cpp
4. Returns an answer grounded in the retrieved text, with the source document named, or a plain "I don't know" if the documents don't cover it

Everything after the initial model download runs offline. No API calls, no cloud dependency, no telemetry.

## Why this exists

I was asked, fairly, to back up some CV claims with something more concrete than a description. Rather than argue the point, I spent a few days building this and measuring what actually happens when you put a small open model to work on Urdu language questions against real documents. Some of what I found supported my assumptions. Some of it didn't. Both are in here.

## Corpus

Eleven public regulation PDFs from Khwaja Fareed University of Engineering and Information Technology: admission, examination, fees, hostel, library, migration/transfer, scholarship, semester, student discipline, and student societies regulations, plus the academic calendar. All public documents, nothing private or institutional in a sensitive sense. Two scanned notification images are also present in the corpus but weren't processed in this pass (see LIMITATIONS.md).

## Architecture

The retrieval layer (embedding plus FAISS search) is what I'd call my own version of what a document retrieval system in this space is typically named. The generation layer is a quantized Qwen3-4B-Instruct model served locally through llama.cpp. Between them sits a small amount of query preprocessing that I tested but didn't end up keeping in the default path, because the evidence didn't support it (see the FAHM section in FINDINGS.md).

```
query
  -> embed (BGE-M3)
  -> retrieve top-k chunks (FAISS)
  -> build prompt with retrieved context
  -> generate answer (quantized Qwen3, local, CPU only)
  -> return answer with cited source
```

## Running it

You'll need Python 3.12, a built copy of llama.cpp, and a converted GGUF model. Rough steps:

```bash
# extract and chunk the corpus
python pipeline/extract.py
python pipeline/chunk.py

# build the retrieval index (one-time cost, independent of which model you generate with)
python pipeline/embed_index.py

# in a separate terminal, serve a quantized model
./llama-server -m models/gguf/qwen3-4b-q4_k_m.gguf -c 3072 --port 8080 --parallel 1

# ask questions
python pipeline/rag.py
```

To run the evaluation harness against a given quantization level:

```bash
python eval/score.py --model-tag q4_k_m
```

## Headline results

Full detail and reasoning is in FINDINGS.md, but the short version:

Running the base model without retrieval produces confident, fluent, completely wrong answers to policy questions it has no way of actually knowing. Adding retrieval grounding fixes this cleanly when retrieval finds the right document, and the model correctly says it doesn't know when the right chunk isn't there, rather than guessing.

Retrieval itself has two separate weak points. One is a plain top-k problem: a document with many chunks on one topic can bury the right answer past a small cutoff, and raising k fixes it. The other is a real gap between how people actually type Roman Urdu and how the source documents are formally worded in English, and this one doesn't go away just by retrieving more chunks.

I tested two ways of closing that gap, a light spelling normalizer and a translation step. Neither reliably solved it. The normalizer measured no benefit at all. Translation helped one case and left another one completely unfixed, while introducing its own translation errors along the way. I think the honest conclusion is that this specific gap needs a better retrieval strategy (a reranker, hybrid search) more than it needs query preprocessing, and I'd want to test that next if I had more time.

Across four quantization levels of the same model, generation quality on a small evaluation set was roughly flat to mildly better at lower precision, which runs against what I expected going in. I don't think ten questions is enough to call that a real pattern rather than noise, and I say so directly in FINDINGS.md. What was unambiguous is that the largest quantization level tested was too slow to be usable on this hardware regardless of any quality question.

## Files

- `pipeline/` — extraction, chunking, embedding, retrieval, and generation code
- `eval/` — the evaluation set, scoring harness, and the two FAHM comparison scripts
- `corpus/` — source PDFs and processed text
- `FINDINGS.md` — the detailed results and reasoning behind them
- `LIMITATIONS.md` — what this project does not cover and why
- `WALKTHROUGH.md` — a full account of the process, including the mistakes

## License

Apache-2.0.
