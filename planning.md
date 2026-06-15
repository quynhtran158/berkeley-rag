# Planning: The Unofficial Berkeley CS Guide

## Domain

**UC Berkeley Computer Science — Student-Generated Course & Campus Knowledge**

The CS department at UC Berkeley publishes syllabi, course descriptions, and official advising pages. What it does NOT publish is the practical, experience-based knowledge that determines whether students succeed: which professor's exam averages 50% vs. 70%, which projects take three days vs. three hours, which dining hall has a 45-minute lunch line, and how to navigate off-campus housing leases without losing a security deposit. This knowledge circulates among students via Reddit (r/berkeley), Rate My Professors, Discord servers, and word-of-mouth — but it is fragmented, buried in comment threads, and inaccessible to incoming students or those who don't know where to look. A RAG system over this domain transforms scattered anecdotes into a searchable, grounded resource.

---

## Documents

| # | File | Source Type | Covers |
|---|------|-------------|--------|
| 1 | `prof_hilfinger_cs61b.txt` | Rate My Professors (scraped + manual) | CS61B under Hilfinger — projects, grading, exam tips |
| 2 | `prof_hug_cs61b.txt` | Rate My Professors + Reddit | CS61B under Hug — comparison to Hilfinger, study tips |
| 3 | `cs61a_tips_reddit.txt` | Reddit r/berkeley | CS61A survival guide — professors, projects, exam strategy |
| 4 | `cs70_reviews.txt` | Rate My Professors + Reddit | CS70 — proof-based exams, professor comparison (Rao vs Ayazifar) |
| 5 | `cs189_ml_reviews.txt` | Rate My Professors + Reddit | CS189 Machine Learning — prerequisites, topic list, exam tips |
| 6 | `cs162_os_reviews.txt` | Rate My Professors + Reddit | CS162 Operating Systems — Pintos projects, exam content |
| 7 | `cs186_databases_reviews.txt` | Rate My Professors + Reddit | CS186 Databases — project structure, exam topics (ARIES, B+ trees) |
| 8 | `cs61c_arch_reviews.txt` | Rate My Professors + Reddit | CS61C Architecture — Dan Garcia, cache problems, projects |
| 9 | `cs170_algorithms_reviews.txt` | Rate My Professors + Reddit | CS170 Algorithms — study list, homework policy, professor comparison |
| 10 | `cs_major_advising_tips.txt` | Reddit r/berkeley wiki | Major declaration, course sequencing, registration tips |
| 11 | `cs_internship_recruiting_tips.txt` | Reddit + student blogs | Recruiting timeline, referrals, LeetCode strategy, salary context |
| 12 | `campus_dining_reviews.txt` | Reddit + Yelp | Dining hall comparison, hours, wait times, best options |
| 13 | `housing_off_campus_reviews.txt` | Reddit r/berkeley | Neighborhoods, lease tips, rent control, specific building reviews |
| 14 | `mental_health_resources.txt` | Reddit r/berkeley | CAPS wait times, peer support, DSP accommodations, study environments |

These sources cover five distinct subtopics (courses, major planning, jobs, campus life, housing) so the system can answer a wide variety of student questions.

---

## Chunking Strategy

**Strategy: Paragraph-aware splitting with fixed-size fallback**

- **Chunk size:** 600 characters (~120–150 words)
- **Overlap:** 100 characters (~20 words)
- **Method:** Split on double newlines (paragraph boundaries) first; if a paragraph exceeds 600 characters, split at 600 with 100-character overlap.

**Reasoning:**

The documents are a mix of structured review-style text (short opinion fragments of 2-4 sentences) and longer informational paragraphs (exam topic lists, neighborhood breakdowns). A pure fixed-size split would slice review text arbitrarily, cutting a sentence like "Exam average is 55% — the curve is generous" in half. Paragraph splitting keeps each opinion unit together.

600 characters is about the length of one review or one section of a guide. This is large enough to contain a complete thought (e.g., a professor's grading style), but small enough to match a specific query (e.g., "what are Hilfinger's exam averages?"). Chunks smaller than ~200 characters would often lack enough context to be meaningful in isolation. Chunks larger than ~1,000 characters would cover too many topics to match precisely.

100 characters of overlap ensures that if key information spans a paragraph boundary (e.g., "The midterm average is 55%" falls at the end of a paragraph and "the curve brings it to B+" starts the next), both chunks contain partial coverage of the fact.

**If chunks are too small:** Retrieval returns fragments that can't stand alone — "Hilfinger's exams are" is not answerable.
**If chunks are too large:** Retrieval matches too broadly — a 2,000-character chunk about a professor's entire course style gets pulled for very specific questions about just one aspect.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`

This model runs locally with no API key and no rate limits. For this project, it is ideal: it produces 384-dimensional vectors, is fast enough to embed 300–500 chunks in under a minute on CPU, and performs well on semantic similarity for English short-text retrieval. It was trained on a diverse corpus that includes Q&A pairs, making it well-suited for matching student questions to review-style answers.

**Top-k:** 5 chunks per query

With k=5, the LLM context window receives ~3,000 characters of relevant material — enough for a complete, grounded answer while staying well within the 8k context window of the generation model. k=3 risks missing relevant information when it's split across chunks; k=8+ dilutes the context with loosely related material.

**Production tradeoffs I would consider:**
- **Context length:** `all-MiniLM-L6-v2` has a 256-token limit per chunk. Longer documents need `nomic-embed-text` or `text-embedding-3-large` (OpenAI). For this dataset, 600-character chunks fit comfortably.
- **Multilingual support:** Not needed for this English-only corpus. If extended to multilingual reviews, `paraphrase-multilingual-MiniLM-L12-v2` would be appropriate.
- **Cost:** Local embedding has zero marginal cost. OpenAI's `text-embedding-3-small` costs $0.02/1M tokens — negligible for this corpus size but meaningful at scale.
- **Domain accuracy:** General models can miss domain-specific vocabulary (e.g., "ARIES" in a database context vs. "Aries" in astrology). A fine-tuned model on Berkeley CS content would improve retrieval, but is out of scope for this project.
- **Latency:** `all-MiniLM-L6-v2` embeds queries in ~20ms on CPU. For a production system with SLA requirements, a GPU-hosted API (like Cohere's Embed API) would provide faster query embedding.

---

## Evaluation Plan

| # | Test Question | Expected Correct Answer |
|---|---------------|------------------------|
| 1 | What is the difference between taking CS61B with Hilfinger vs. Hug? | Hug is more scaffolded, provides clearer specs, and has slightly higher exam averages (~60-70%). Hilfinger is harder, cryptic specs, lower exam averages (~50-60%). Most students recommend Hug unless they want maximum rigor. |
| 2 | What are the exam topics I should study for CS186? | SQL (complex queries), relational algebra, B+ trees (insertions/deletions), buffer management (LRU/MRU/CLOCK), two-phase locking, and ARIES recovery protocol. ARIES appears on every exam. |
| 3 | Which dining hall has the shortest wait times? | Clark Kerr and Foothill are consistently less crowded. Avoid Crossroads during 12pm-1pm lunch rush, which has the longest wait times (10-15 min even off-peak, worse during peak). |
| 4 | When should I start applying for software engineering internships if I want a summer internship at Amazon? | Amazon opens applications in September for the following summer. Online assessments go out October-November, offers by December. Apply immediately when applications open. |
| 5 | How long is the wait at CAPS for a mental health appointment? | Typically 2-4 weeks for the first appointment. For same-day slots, call at 8am. For emergencies, call the Tang Center crisis line (510-642-9494), not CAPS. |

---

## Anticipated Challenges

1. **Information split across chunk boundaries:** Some facts span two sentences in different paragraphs (e.g., "Exam average is 55%" in one paragraph, "the curve brings it to B+" in the next). The 100-character overlap mitigates this but doesn't eliminate it. The evaluation may reveal cases where the answer is partially retrieved but incomplete.

2. **Ambiguous queries returning off-topic chunks:** If a user asks "how hard is the CS department?" this could pull from exam reviews, mental health resources, housing stress, and internship difficulty simultaneously. The LLM must synthesize diverse context chunks coherently or acknowledge the ambiguity. Off-topic retrieval is most likely for broad, unspecific questions.

3. **Source attribution confusion:** Multiple documents cover overlapping topics (both `prof_hilfinger_cs61b.txt` and `prof_hug_cs61b.txt` discuss CS61B). The generation step must cite specific source files, not just "the documents."

---

## AI Tool Plan

| Pipeline Component | What I'll Give the AI | What I Expect It to Produce |
|---|---|---|
| `ingest.py` — document loading & chunking | This Chunking Strategy section + list of document filenames + sample document text | A script with `load_documents()` and `chunk_text()` functions implementing paragraph-aware 600-char/100-overlap splitting |
| `embed.py` — ChromaDB setup & embedding | Architecture diagram (below) + Retrieval Approach section + sample chunk structure | Code to initialize ChromaDB, embed chunks with `all-MiniLM-L6-v2`, store with source metadata |
| `query.py` — retrieval + generation | Architecture diagram + grounding requirement (answer from retrieved context only, cite sources) + Groq API info | `retrieve()` and `generate()` functions with a grounding system prompt and source attribution |
| `app.py` — Gradio interface | Gradio skeleton (from project spec) + my `query.py` function signature | A working Gradio `gr.Blocks()` app wired to my `ask()` function |
| `evaluate.py` — evaluation runner | My 5 test questions with expected answers + the `ask()` function interface | A script that runs all 5 questions, prints responses, and outputs a structured evaluation report |

I will review all generated code for: (a) grounding enforcement in the system prompt, (b) correct ChromaDB collection operations, (c) proper source metadata handling. I will not accept generated code that I cannot explain line by line.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION                           │
│  Load .txt files from /documents/ → clean whitespace/artifacts  │
│  Tool: Python built-in (open/read) + custom cleaner              │
└─────────────────────────┬────────────────────────────────────────┘
                          │ raw cleaned text
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       CHUNKING                                   │
│  Split on paragraph boundaries (double newline)                  │
│  Fallback: fixed 600-char chunks with 100-char overlap           │
│  Tool: Custom Python (no external library needed)                │
└─────────────────────────┬────────────────────────────────────────┘
                          │ list of (chunk_text, source, chunk_idx)
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                  EMBEDDING + VECTOR STORE                        │
│  Embed each chunk → 384-dim vector                               │
│  Store in ChromaDB collection with source metadata               │
│  Embedding model: all-MiniLM-L6-v2 (sentence-transformers)       │
│  Vector store: ChromaDB (local, persistent)                      │
└─────────────────────────┬────────────────────────────────────────┘
                          │ indexed vector store
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL                                  │
│  Embed user query → semantic similarity search                   │
│  Return top-5 chunks + source filenames                          │
│  Tool: ChromaDB query() + all-MiniLM-L6-v2                       │
└─────────────────────────┬────────────────────────────────────────┘
                          │ retrieved chunks + sources
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                       GENERATION                                 │
│  Construct prompt: system (grounding instruction) + context      │
│  Generate grounded answer with source attribution                │
│  LLM: Groq llama-3.3-70b-versatile                               │
│  Tool: groq Python client                                        │
└─────────────────────────┬────────────────────────────────────────┘
                          │ answer + source list
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                     QUERY INTERFACE                              │
│  Input: text question from user                                  │
│  Output: answer + "Retrieved from:" source list                  │
│  Tool: Gradio (gr.Blocks web UI, localhost:7860)                 │
└──────────────────────────────────────────────────────────────────┘
```
