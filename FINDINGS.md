# Findings

This is the detailed results section. For the short version, see the README. For the story of how these were arrived at, including the mistakes, see WALKTHROUGH.md.

## 1. Ungrounded generation fabricates confidently

Before any retrieval was wired up, I asked the base model (Qwen3-4B-Instruct, Q4_K_M, no context) a question in Roman Urdu about what happens if a student misses an exam paper. It answered fluently and at length, with structure and confidence, about NEET, JEE, UPSC and SSC exam repeat policies. These are Indian entrance exams. The actual corpus is Pakistani university regulations and has nothing to do with any of them.

There was no hedging in the answer, no signal that it was guessing. It read exactly like a correct answer. That's the core problem retrieval grounding exists to solve, and it's a much stronger thing to have measured myself than to have read about.

## 2. Grounding fixes fabrication, but only when retrieval actually works

The same question, run through the full pipeline with retrieval turned on, produced a much more honest failure at first: "I don't have that information in the provided context." No fabrication, but also not the right answer, because the retrieved chunks (best similarity score 0.459) were mostly about library procedures and discipline hearings, not the actual exam makeup policy.

Manually checking the source text confirmed the correct policy exists in the corpus, word for word, in a section called the Makeup Exam Policy. The relevant chunk simply wasn't in the top five results returned by retrieval.

Manually inspecting the ranked results with a wider search window found the correct chunk sitting at rank nine. It existed in the index and was reachable, just not within the default cutoff. Raising the retrieval cutoff from five to twelve results fixed this specific case: the same question in English now retrieved the right chunk and the model answered correctly, citing the conditions, the approval process, the deadline, and the fee.

Running the same fix on the original Roman Urdu phrasing of the question did not work. Even at the higher cutoff, the correct chunk still wasn't retrieved, and the similarity scores across the board were noticeably lower than the English version. This showed two separate problems were stacked on top of each other rather than one problem with two symptoms: a retrieval depth issue, which raising the cutoff fixes, and a language gap issue, which it does not.

## 3. Quantization ladder

Given the time available, I tested four quantization levels of the same 4B model rather than the full six-level ladder, chosen to spread across the precision range: Q8_0 (closest to full precision), Q5_K_M, Q4_K_M, and Q4_0 (most aggressive). All four were built from the same source weights and run through the same evaluation set of ten questions, six of which are scored on gold-fact matching and four on whether the model correctly declines to answer something outside the corpus.

| Quantization | File size | Fact-match score | Correct refusal rate | Generation speed |
|---|---|---|---|---|
| Q8_0 | 4.0GB | 0.44 | 1.00 | Impractically slow on this hardware, multiple queries took 15 to 40 minutes |
| Q5_K_M | 2.7GB | 0.44 | 1.00 | Normal |
| Q4_K_M | 2.4GB | 0.50 | 1.00 | Normal, roughly 5 tokens per second baseline |
| Q4_0 | 2.3GB | 0.56 | 0.75* | Fast |

\* The Q4_0 refusal score needs unpacking. On the question about Friday holidays, the model gave a direct, confident answer ("No, Friday is not a holiday") rather than a hedged "not mentioned in context" response the automated scorer was looking for. Checked against the actual corpus and against reality, this answer is correct and arguably better than the more hedged phrasing the other quantization levels gave on the same question. I'm not counting this against Q4_0. But I'm also not treating it as proof that the smallest quantization is generally more accurate. It's one data point on one easy question, and I'd want a much bigger evaluation set before drawing a real conclusion either way.

Taking the fact-match scores at face value, quality holds roughly flat to slightly better as precision drops, which is the opposite of what I expected going in and the opposite direction from what published research on quantization and low-resource languages generally finds. With only six scoreable questions, I don't think this difference is large enough to call a real pattern. The number I'd actually stand behind from this ladder is the practical one: Q8_0, despite the highest precision, was not usable in any real sense on this hardware. Two individual questions took between fifteen and forty minutes to generate a response. That alone rules it out for a client deployment on similar hardware regardless of any accuracy question.

## 4. A generation-quality failure separate from retrieval

On the Roman Urdu paper-repeat question at Q4_K_M, the model's answer degraded into a repetition loop partway through, the word "koi" repeated roughly thirty times before the response ended. This happened independently of the retrieval problem already documented above. It's a distinct failure mode worth naming on its own: even once retrieval is fixed, quantized generation on Roman Urdu output can still degrade in ways that don't show up on English prompts.

## 5. Scoring the evaluation set correctly took more than one pass

The first version of the automatic scorer checked for gold facts using plain English word matching. Every single answerable question scored zero, not because the answers were wrong, but because the model was correctly answering in Roman Urdu and mixed language, exactly as instructed, and the scorer only recognized English phrasing. Rewriting the scorer to use embedding similarity instead of literal word matching fixed this, and re-checking the same saved answers against the new scorer brought the Q4_K_M baseline to a mean fact-match score of 0.50.

A related issue turned up in refusal detection specifically. Q8_0's refusal rate initially looked worse than Q4_K_M's, which was surprising given it's the higher precision model. The actual answer in question was a correct refusal, stated in Roman Urdu ("koi information nahi hai"), that the English-only refusal keyword list simply didn't recognize. Adding Roman Urdu markers to the refusal check fixed this and brought Q8_0's refusal rate back in line.

Both of these were the same underlying mistake in different places: building evaluation tooling with an unstated English-only assumption for a system that's explicitly bilingual. I think that's a genuinely useful lesson rather than just an implementation detail. Every piece of scoring infrastructure for a bilingual system needs the same scrutiny as the system it's evaluating, or it will produce confident, wrong numbers that look completely plausible until someone checks them by hand.

## 6. The evaluation set itself needed correcting

Building the gold-fact evaluation set involved retrieving candidate chunks for each collected query and manually deciding whether the right answer was present. On the first pass, five of ten queries came back as "not found in the retrieved candidates." Cross-checking each one directly against the source text found that three were genuine gaps (the corpus simply doesn't cover Friday holidays, bus schedules, or gym fees) but two were retrieval misses where the content did exist and just wasn't surfaced. Both were corrected once identified.

A third correction turned up later during result review: a question about project funding had been given a gold fact drawn from a general financial assistance clause that, on closer reading, wasn't really about the same thing the question was asking. The corpus only covers scholarships, not project-specific funding, and the model's refusal was actually the right answer. That entry was relabeled.

None of this is a criticism of the process, it's the process working as intended. But it's worth stating plainly: a "not found" label from a tool that shares the same retrieval weaknesses as the system under test cannot be trusted at face value. Out of five initial "not found" labels, two turned out to be wrong, a meaningful error rate that would have quietly understated the system's real capability if left unchecked.

## 7. FAHM: closing the Roman Urdu gap

The retrieval gap between casual Roman Urdu phrasing and the corpus's formal English wording (see finding 2) motivated two lightweight experiments, tested on the six evaluation questions with known correct source chunks.

**Spelling normalization.** A small, hand-built list of common Roman Urdu spelling variants (kya/kia/kiya, sakti/skti, and similar) was applied to each query before embedding, and the retrieval rank of the correct chunk was compared before and after.

| Query | Raw rank | Normalized rank |
|---|---|---|
| Paper repeat (known miss) | not found | not found |
| Fees installment | 1 | 1 |
| Ragging/bullying | 1 | 2 |
| Library fee | 1 | 1 |
| Event hosting (known miss) | not found | not found |
| Teacher complaint | 1 | 2 |

Zero improvements, two cases slightly worse, four unchanged. Normalization didn't fix either of the two genuine retrieval misses. My read on this is that BGE-M3 already handles minor spelling variation reasonably well on its own, so a manual normalizer adds no real signal while occasionally introducing small noise.

**Translation.** Before building the second approach, I checked whether the corpus contained any Urdu-script text at all, since the original plan called for transliterating queries into Urdu script. It doesn't; every document in the corpus is written in English. Transliterating a query into Urdu script would move it further from the corpus's actual language rather than closer, so I adapted the approach to translate each query into English instead, using a short call to the same local model, and re-ran the same rank comparison.

| Query | Raw rank | Translated rank |
|---|---|---|
| Paper repeat (known miss) | not found | 20 |
| Fees installment | 1 | 1 |
| Ragging/bullying | 1 | 2 |
| Library fee | 1 | 1 |
| Event hosting (known miss) | not found | not found |
| Teacher complaint | 1 | 2 |

One case moved from completely unfound to rank twenty, which is technically an improvement but not a practically useful one at any reasonable retrieval cutoff. The event hosting question stayed unfound even with a translation that read correctly to me on inspection ("Can a class event be organized? If yes, where?"), because the corpus actually frames that topic around funding approval through a specific office rather than general feasibility, and a correct translation doesn't fix a framing mismatch.

Translation also isn't free of its own errors. An early test of the translation step rendered "if a paper is missed, can it be repeated" as "if the paper is lost, can we get a duplicate copy," conflating a missed exam with a lost document. The version used in the final comparison run was better, but this is a real risk worth naming rather than glossing over.

**Combined read.** Neither approach reliably closed the gap. Spelling normalization measured no benefit. Translation measured one marginal improvement and left one case completely unfixed, while introducing its own error mode. I think the honest conclusion is that the actual gap here isn't primarily a spelling or vocabulary problem, it's a mismatch between how people ask questions casually and how institutional documents are formally worded, and that's a retrieval ranking problem more than a query preprocessing problem. A reranking step over a wider candidate pool, or a hybrid retrieval approach combining keyword and semantic search, seem like more promising directions, but neither was built in this pass.

## 8. Hardware and operational findings

A few things turned up along the way that are less about the model and more about actually running this stack on constrained hardware:

Running the model at full FP16 precision (8GB of weights on an 8GB machine) made the system unresponsive rather than just slow. This wasn't a graceful degradation, it was a genuine freeze requiring a hard reset in one case, most likely from swap thrashing with no memory headroom left for anything else.

A crash partway through the first scoring run turned out not to be a kernel level out-of-memory kill (checked directly through the system journal), but severe resource contention from running the generation server and the scoring script's own embedding model at the same time on 8GB of RAM. The fix that helped was reducing the context window on the generation server and closing other running applications.

Looking at the raw server logs from the quantization runs turned up something specific: llama-server reserves four parallel inference slots by default, each with its own memory reserved for the full context window, even though this project only ever sent one request at a time. That's roughly four times the memory actually needed sitting reserved throughout every run, and it's a plausible explanation for more than one of the slowdowns and freezes encountered along the way, including one query that dropped to roughly one twentieth of its normal generation speed for no other apparent reason. Starting the server with a single parallel slot instead of the default four should meaningfully reduce this, though it wasn't re-tested after being identified.

None of this is really about model quality. It's the kind of thing that matters once you're trying to actually run something like this at a client site on hardware that isn't a fresh, high-spec machine, and I think it belongs in this document as much as any of the accuracy numbers do.
