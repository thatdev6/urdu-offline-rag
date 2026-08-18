# Walkthrough

This is a full account of building this project: what I did, why I did it that way, what went wrong, and how I got past it. I wanted to write this down honestly rather than cleaning it up into a straight success story, because I think the mistakes and how they got caught say more about how I work than a tidy final result would on its own.

## Why I built this

I'd made some claims on my CV about experience relevant to this kind of work, and rather than argue the point back and forth, it seemed more useful to spend a few days actually building something and measuring what happened. The plan going in was to build an offline, bilingual retrieval system against real documents, run a small local language model against it entirely on CPU, and see what broke. Something breaking wasn't a risk to the plan, it was the point of the exercise. I wanted numbers I'd taken myself rather than assertions I was making secondhand.

## Setting up

I started with a Fedora machine, an old i5-3570 from around 2012 with 8GB of RAM and no dedicated GPU for the CPU work (there's an AMD card in the machine, discovered later through the system logs, but the language model inference stayed CPU-only on purpose, to match a realistic no-GPU client deployment rather than take a shortcut on my own hardware). This constraint shaped a lot of what came after, more than I expected going in.

The early setup work was mostly ordinary but ran into a string of small, real problems worth recording because each one cost real time and each fix generalizes.

The system's `/tmp` folder turned out to be a small RAM-backed filesystem, capped well under 4GB, and large pip installs kept failing with a disk quota error that had nothing to do with actual disk space, since the real disk had well over a hundred gigabytes free. The fix was pointing the temp directory at a folder inside the project itself instead of the system default.

The first Python virtual environment got created under Python 3.14 by accident, since that's what the system's default `python3` resolved to. This broke `numpy`, which llama.cpp's conversion tooling pins to an older version that has no prebuilt wheel for that new a Python release, and the fallback to building from source failed outright because there was no C compiler installed at all. Installing `gcc` and switching explicitly to Python 3.12 fixed the immediate problem, but a second issue showed up right behind it: llama.cpp's own tooling wants `numpy` under 2.0, while the retrieval pipeline's dependencies (through `scikit-learn`) want `numpy` 2.0 or newer. Trying to satisfy both in the same environment broke one or the other every time. The actual fix was giving up on one shared environment and splitting into two: one for the conversion tooling, one for everything else. Not elegant, but reliable, and it's the kind of fragmentation that's worth knowing about going in rather than discovering halfway through a session.

I also set up GitHub CLI authentication, which is a one-time step per machine using a browser device code flow, and built llama.cpp from source with cmake once a compiler was actually available.

## Getting a working model

The plan was to test two model sizes, a 4B and a 1.7B, both from Qwen3's instruction-tuned line. The first attempt at the 1.7B model grabbed the wrong Hugging Face repository entirely. It turned out to be a reranking model, not a text generation model, and llama.cpp correctly refused to convert it once I tried. The fix in the moment was simple, check the `architectures` field in the model's config file before downloading anything, but I didn't catch it until after the download, and the 1.7B never actually got revisited given everything else that came up. The 4B model download itself also needed resuming twice after incomplete first passes before all the weight shards were actually present.

Converting the downloaded weights into the GGUF format llama.cpp uses worked cleanly once the model itself was correct, and I made a point of understanding what that conversion step actually does rather than just running the command: it's a format change, not a precision change, the weights stay the same, they're just repackaged into a container llama.cpp's runtime can read directly.

Before doing anything else, I tried running the model at its full, unquantized precision directly on the 8GB machine, mostly out of curiosity about where the real ceiling was. The system froze. Not slowed down, actually became unresponsive, almost certainly from having no memory headroom left once an 8GB model, the operating system, and everything else all had to share 8GB of RAM. Quantizing down to Q4_K_M, at roughly a quarter of the file size, fixed this immediately and became the model I used for most of the early work.

## First look at how the model handles Urdu

Before building any retrieval, I ran a handful of mixed Roman Urdu and Urdu-script prompts straight at the quantized model with no context at all, just to see whether it was even worth continuing down this path. The language handling itself was genuinely solid, no script confusion, no garbling, fluent replies in whichever register the question was asked in.

What stood out instead was a hallucination. Asked in Roman Urdu what happens if a student misses an exam, the model answered fluently and confidently with details about Indian entrance exams, NEET, JEE, UPSC, none of which have anything to do with the actual Pakistani university this project is based on. No hedging, no signal it was guessing. That one exchange became the clearest single piece of evidence for why grounding the model in real documents mattered, and I kept coming back to it throughout the rest of the project as the before picture.

## Building retrieval

The corpus is eleven public regulation PDFs from a real Pakistani university, covering admissions, exams, fees, hostel rules, library rules, discipline, and student societies. I wrote a small extraction script using `pdfplumber` that also flags any page with suspiciously little extracted text, since scanned documents or documents with a broken text layer can silently produce garbage instead of real content. All eleven PDFs came back clean, with proper text throughout, which was itself worth noting rather than assuming: these are official, born-digital documents, not scans, and that result doesn't say anything about the two scanned notification images also sitting in the corpus folder, which needed a completely different processing path that I didn't get to.

Chunking split the cleaned text into roughly two hundred word segments with some overlap between them, sized conservatively on purpose, since Urdu text tends to carry less content per token than English does, and a chunk size tuned for English tends to hold noticeably less actual Urdu content than the token count would suggest.

For the retrieval layer itself I used BGE-M3, a multilingual embedding model, and built a FAISS index over the chunked corpus. This is the part of the pipeline that does the actual document search, and it's worth naming clearly since the company I was preparing this for uses their own internal name for this same kind of layer; building my own reference version of it and being able to describe it in their terms directly felt like a stronger way to talk about the work than generic RAG language would have been.

## Wiring generation to retrieval

Once retrieval was working, I built the piece that actually ties a question to an answer: embed the query, pull back the most relevant chunks, build a prompt instructing the model to answer only from what's given, match the language and script of the question, skip the markdown and emoji styling the model defaults to, cite its source, and say plainly when it doesn't know rather than guess.

Re-running the exact hallucination question from earlier through this full pipeline produced a much more honest result: the model correctly said it didn't have that information, rather than fabricating an answer. That was progress, but it also wasn't the real fix yet, because the retrieved chunks weren't actually the right ones. Manually checking the source text found the real policy, called the Makeup Exam Policy, sitting in the corpus in almost the same words as the original question. It just hadn't come back in the top results.

Widening the search and looking at where that chunk actually ranked found it sitting at position nine, just past the cutoff I'd been using. Raising the retrieval cutoff fixed the English version of the question outright, the model now retrieved the right chunk and gave a correct, detailed answer. Running the same fix on the original Roman Urdu phrasing of the same question did not work. The correct chunk still wasn't found, and the similarity scores were noticeably weaker across the board. That told me two separate problems had been tangled together the whole time: a plain retrieval depth issue, which a bigger cutoff fixes, and a real language gap between casual Roman Urdu and the corpus's formal English wording, which a bigger cutoff does nothing for.

## Building the evaluation set

I collected ten real questions from people I know, asked in their own natural Roman Urdu spelling rather than corrected or cleaned up. Rather than manually searching eleven documents by hand for each one, I built a small interactive tool that retrieves candidate chunks for each query and lets me mark which ones actually answer it, along with two to four short facts the correct answer needs to contain.

On the first pass, half the queries came back as not answerable from the retrieved candidates. Rather than accepting that number, I cross-checked each one directly against the source text, the same way I'd already done for the exam policy case. Three genuinely weren't covered by the corpus at all, Friday holidays, bus schedules, and gym fees simply aren't in these documents. But two were retrieval misses, not real gaps, including the same exam policy question from before and a separate question about hosting a class event, both of which had real answers sitting in the corpus that just hadn't surfaced. I corrected both entries once I found this. A fifth issue turned up later, during result review rather than during the initial build: a question about project funding had been given a gold fact that, on a closer look, wasn't really about the same thing being asked, and the model's answer declining to help was actually the correct one. That got relabeled too.

I don't think any of this reflects poorly on the process. It's what building a trustworthy evaluation set for a system like this actually looks like. But it's worth being direct about it rather than presenting the final set as if it had been right from the start.

## Scoring, and getting it wrong before getting it right

The scoring script runs the evaluation set through the live pipeline and checks whether each answer contains the gold facts it's supposed to, plus separately checks whether the model correctly declines on questions the corpus doesn't cover. The first version checked for gold facts using plain English word matching. Running it produced a flat zero across every single answerable question, which was alarming until I actually read the answers themselves and realized they were correct, just written in Roman Urdu, exactly as the system prompt asked for. An English-only word matcher was never going to recognize that. I rewrote the scorer to compare meaning using the same embedding model already doing the retrieval work, rather than matching literal words, and re-checked the saved answers against the new version rather than re-running the whole pipeline from scratch.

A related version of the same mistake showed up again later, in a different part of the same script. One quantization level scored worse on refusal correctness than the others, which was surprising since it was the higher precision model. The actual answer in question was a correct, honest refusal, just stated in Roman Urdu, and the refusal detection list was, again, English only. Same root cause in a different place: I'd built evaluation tooling with an unstated assumption that answers would be in English, for a system whose entire point is that they wouldn't be.

I also had the scoring script crash partway through a run, which cost an entire batch of results the first time it happened, since the original version only saved its output once everything had finished. I rewrote it to save after every single question instead, and to skip anything already scored on a rerun, so a future crash would only cost the one question in progress rather than the whole batch. That fix mattered more than it might sound like, given how often memory pressure on this machine caused things to slow down or lock up over the course of the project.

## The quantization ladder

With scoring actually working correctly, I built out four quantization levels of the same model, spread across the precision range rather than the full six level ladder the original plan called for, given the time actually available. Building each one was quick, pure CPU work with no memory risk. Running the evaluation set against each one was slower and, at the highest precision level tested, genuinely impractical, with individual questions taking between fifteen and forty minutes before finishing.

The scores themselves came out roughly flat across the four levels, with a very slight edge toward the smaller, more aggressively quantized models, which runs against what I expected and against the general direction published research on this topic tends to find. I don't think ten questions is enough to call that a real trend rather than noise, and I said so directly rather than rounding it up into a stronger claim than the evidence supports. The clearer, more confident result from this stage was practical rather than statistical: the largest quantization level tested simply wasn't usable at any reasonable response time on this hardware, independent of whatever accuracy it might have had.

Reading through the raw server logs from these runs afterward turned up something worth fixing going forward: the server was reserving four parallel processing slots by default, each with its own memory set aside for the full context window, even though only one request was ever being sent at a time. That's a real, avoidable source of memory pressure, and very likely connected to more than one of the freezes and slowdowns from earlier in the project, including one question that dropped to a small fraction of its normal generation speed for no clear reason at the time.

## Testing whether normalization actually helps

The Roman Urdu retrieval gap found earlier was the clearest open question left, so I tested two lightweight ways of closing it, using the six evaluation questions with known correct source chunks and checking, before and after, what rank the correct chunk came back at.

The first was a small, hand-built list of common Roman Urdu spelling variants, applied to the query text before embedding. This made no measurable difference. Nothing improved, and two cases actually got slightly worse. My read on this is that the embedding model likely already handles small spelling variation well enough on its own, so a manual normalizer on top of it adds no real signal and just occasionally introduces small noise.

Before building the second approach, which the original plan called for as transliterating the query into Urdu script, I checked whether the corpus actually contained any Urdu-script text to retrieve against. It doesn't, every document in this corpus is written entirely in English. Transliterating a query into a script the corpus doesn't use at all would have moved it further from the target, not closer, so I adapted the approach into translating the query into English instead, using a short call to the same local model rather than a heavier dedicated translation pipeline.

This helped a little, and not enough to call it a fix. One previously unfound case moved to the very edge of a wide search window, technically an improvement but not a practically useful one. A second unfound case stayed unfound entirely, even though the translation itself read correctly, because the corpus frames that particular topic differently than the question assumed, funding approval through a specific office rather than general feasibility. Along the way I also caught the translation step making a real error on one query, rendering a question about a missed exam as a question about a lost document, which is a useful reminder that translation isn't a free preprocessing step either.

Put together, neither approach reliably solved the underlying problem. My honest conclusion is that the actual gap here is less about spelling or vocabulary and more about the mismatch between how people casually phrase a question and how a formal institutional document is worded, and that feels like a ranking and retrieval problem more than something query preprocessing alone is going to fix. A reranking step or a hybrid keyword-plus-semantic search seem like the more promising next things to try, but I didn't get to build either given the time left.

## Where I stopped

At this point I had a working end-to-end pipeline, a real evaluation set that had been corrected through actual use rather than assumed correct, a four-level quantization comparison with honestly caveated results, and two tested approaches to the Roman Urdu gap with a clear, evidence-based answer about what does and doesn't help. OCR for the two scanned notification images, a proper deployment sizing table, and a second model size comparison were all identified as reasonable next steps but weren't completed, mostly a matter of time rather than not knowing what to do next.

## Tools and technologies used

Python 3.12, llama.cpp built from source for model conversion, quantization, and serving, Qwen3-4B-Instruct as the generation model, BGE-M3 for embeddings, FAISS for the vector index, pdfplumber for PDF text extraction, sentence-transformers as the embedding interface, and a local llama-server instance exposing an OpenAI-compatible API for the pipeline to call. Everything ran on a single Fedora machine with no GPU involvement in the language model inference and no external API calls of any kind after the initial model download.

## What I'd do differently

I'd build the memory-pressure fixes in earlier rather than discovering them through repeated crashes. The single parallel slot setting on the server especially, that's a five second change that would have saved real time if I'd known about it from the start rather than finding it in the logs after the fact. I'd also build the bilingual awareness into the scoring script from the very first version rather than as a fix after seeing a suspicious zero score, since in hindsight it was an obvious thing to get wrong for exactly the kind of system this is. Neither of these change the actual findings, but they'd have gotten me to the findings faster.
