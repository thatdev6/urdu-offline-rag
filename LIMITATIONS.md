# Limitations

This is a personal, time-boxed project, not a production system. Here is what it doesn't cover and where the numbers should and shouldn't be trusted.

## Hardware

Testing was done on an i5-3570 (2012, no AVX2 support) with 8GB of RAM and no GPU. This is below what's typically recommended even for a 4B model at full precision, and that showed up directly: attempting to run the model at FP16 made the system unresponsive rather than just slow. Every practical result in this project comes from quantized models running on genuinely constrained hardware, not a best-case setup.

Because of this, some of the numbers here are pessimistic compared to what you'd see on a modern laptop. Generation speed at Q4_K_M averaged around 5 tokens per second in short prompts and dropped further once realistic retrieval context was added. A more current CPU, particularly one with AVX2 or AVX-512 support, would likely do noticeably better. I'd treat the timing numbers in this project as a lower bound for old hardware, not a general expectation.

The 8GB memory ceiling also caused a few operational problems along the way: a system freeze at FP16, at least one severe slowdown likely caused by llama-server reserving more parallel inference slots than were actually being used, and repeated timeouts on the largest quantization level tested. These are documented in FINDINGS.md and I think they're genuinely useful data points about running this kind of system on constrained hardware, but they mean some of the quant ladder results (particularly Q8_0) come with real caveats about practicality rather than pure accuracy comparison.

## Corpus scope

The corpus is eleven regulation PDFs, all entirely in English. There is no Urdu-script content anywhere in the source documents, which is worth stating plainly because it shaped how the FAHM experiments were designed (see FINDINGS.md). Some topics that came up in the collected query set, Friday holidays, bus transport schedules, gym fees, simply aren't covered by these documents at all. That's a real property of this university's public regulation set, not a gap in the extraction pipeline.

Two scanned notification images exist in the corpus folder but were not processed. OCR was scoped out of this pass given time constraints. If they mattered for a real deployment, that work is straightforward to pick up but genuinely wasn't started here.

## Evaluation set size

The evaluation set has ten queries, six of which are scoreable fact-match questions. That's small. It's enough to catch clear, structural problems (a chunk that never gets retrieved, a model that fabricates when ungrounded), but it is not enough to make confident statistical claims about which quantization level is genuinely better on quality. Where the results looked close or counterintuitive, I've tried to say so directly rather than round it up to a bigger claim than the data supports.

The queries themselves came from a small, informal group of people I know, asked to type questions the way they normally would rather than in careful, correct spelling. That's useful for realism but it's a narrow sample, and the vocabulary likely reflects the handful of example questions given as prompts more than it reflects the full range of ways people actually phrase these questions.

## Scoring methodology

The automatic scorer went through several rounds of fixing during the project, and I think it's worth being upfront about that rather than presenting the final numbers as if they were correct from the start. Early versions checked for gold facts using literal English word matching, which produced a false zero score across every answerable question, because the model was correctly answering in Roman Urdu and the scorer only understood English. A similar issue existed in the refusal detection, which also started out English-only. Both were fixed by switching to semantic similarity matching and adding Roman Urdu markers respectively, and I re-scored earlier results after each fix. The final numbers in FINDINGS.md reflect the corrected scorer, but any reader should know the methodology wasn't right on the first attempt.

The scoring approach itself, semantic similarity against a handful of gold facts per question, is a reasonable proxy but not a precise one. It won't catch every subtlety in an answer and it can occasionally reward a technically-present phrase without the surrounding meaning being fully correct.

## FAHM experiments

Two lightweight approaches were tested for closing the gap between casual Roman Urdu queries and the formally worded English corpus: a manually curated spelling normalizer, and a translation step using the same local model. Neither was a full solution. The normalizer is a small hand built list of variants, not a proper phonetic algorithm. The translation step used a single small model with a short prompt, and it demonstrably made translation errors on at least one query during testing. Neither approach was tested at scale beyond the six answerable queries in the evaluation set.

A hybrid retrieval approach or a reranking step over a wider candidate pool were identified as more promising directions based on the results, but weren't built or tested given the time available.

## Model swap

An earlier attempt to test a smaller 1.7B model failed because the wrong Hugging Face repository was downloaded (a reranker model rather than an instruction-tuned generation model). This was caught before any wasted evaluation time, but the smaller model was never actually re-downloaded and tested in this pass. All quantization and evaluation results in this project are for the 4B model only.

## What would change with more time

A larger evaluation set, ideally closer to the forty-question range, would let the quantization comparison say something more confident than "flat to slightly favoring smaller quants, possibly noise." A reranking step over a wider retrieval pool seems like the more promising fix for the Roman Urdu retrieval gap than anything tested here. The two scanned notification images would need an actual OCR pass. And the deployment sizing numbers here are drawn from one specific old CPU; a proper sizing table would want at least one more modern reference point for comparison.
