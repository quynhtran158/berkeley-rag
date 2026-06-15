# The Unofficial Berkeley CS Guide 🐻

This is my Week 1 RAG (Retrieval-Augmented Generation) project for AI201. I built a system that lets you ask questions about UC Berkeley CS courses, campus life, housing, and internships — and get answers grounded in real student-generated content from Reddit, Rate My Professors, and student blogs.

The idea came from a simple frustration: the official CS department pages tell you almost nothing useful. What exam average should you expect in CS61B with Hilfinger? Which dining hall has a 45-minute line at noon? How early do you actually need to apply for Jane Street? That knowledge exists — it's scattered across Reddit threads and Discord servers. I pulled it together, chunked it, embedded it, and made it queryable.

---

## How to Run This

You'll need a free Groq API key. Get one at [console.groq.com](https://console.groq.com) — it's free and takes two minutes.

```bash
# 1. Clone and navigate into the project
git clone <your-fork-url>
cd week_1

# 2. Set up your environment
#    On Apple Silicon Mac (recommended):
conda create -n berkeley_rag python=3.11 -y
conda activate berkeley_rag
pip install sentence-transformers==3.4.1 "numpy<2"
pip install chromadb groq python-dotenv gradio

#    On Linux/Windows or Intel Mac:
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# Open .env and paste your key: GROQ_API_KEY=your_key_here

# 4. Build the vector store (run this once)
#    Downloads ~90MB model on first run and embeds all 14 documents
python embed.py

# 5. Launch the web interface
python app.py
# Open your browser to: http://localhost:7860
```

To run the evaluation suite:

```bash
python evaluate.py
```

---

## What You Can Ask

The system only answers from its 14 source documents — it won't make things up or pull from general knowledge. Some good questions to try:

- "What is the difference between taking CS61B with Hilfinger vs. Hug?"
- "What exam topics should I study for CS186?"
- "Which dining hall has the shortest wait times?"
- "When should I start applying for Amazon internships?"
- "How long is the wait for a CAPS appointment?"

If the documents don't have enough information to answer, the system says so directly instead of guessing.

---

## Project Structure

| File | What it does |
|------|-------------|
| `ingest.py` | Loads and chunks the 14 source `.txt` files |
| `embed.py` | Embeds chunks using `all-MiniLM-L6-v2` and stores them in ChromaDB |
| `query.py` | Retrieves top-k chunks and calls Groq's LLM for a grounded answer |
| `app.py` | Gradio web UI — question input, answer output, source citations |
| `evaluate.py` | Runs 5 test queries and compares responses against expected answers |
| `data/` | The 14 `.txt` source documents |
| `chroma_db/` | Persisted ChromaDB vector store (created when you run `embed.py`) |

---

## Document Sources

Domain: UC Berkeley CS — student-generated course and campus knowledge.

| File | Source | Covers |
|------|--------|--------|
| `prof_hilfinger_cs61b.txt` | Rate My Professors + manual collection | CS61B with Hilfinger — exams, projects, grading |
| `prof_hug_cs61b.txt` | Rate My Professors + Reddit r/berkeley | CS61B with Hug — comparison to Hilfinger, study tips |
| `cs61a_tips_reddit.txt` | Reddit r/berkeley | CS61A survival guide — professors, projects, exam strategy |
| `cs70_reviews.txt` | Rate My Professors + Reddit | CS70 — proof exams, professor comparison |
| `cs189_ml_reviews.txt` | Rate My Professors + Reddit | CS189 Machine Learning — prerequisites, exam topics |
| `cs162_os_reviews.txt` | Rate My Professors + Reddit | CS162 OS — Pintos projects, exam content |
| `cs186_databases_reviews.txt` | Rate My Professors + Reddit | CS186 Databases — project structure, ARIES, B+ trees |
| `cs61c_arch_reviews.txt` | Rate My Professors + Reddit | CS61C Architecture — Dan Garcia, cache problems |
| `cs170_algorithms_reviews.txt` | Rate My Professors + Reddit | CS170 Algorithms — study list, homework policy |
| `cs_major_advising_tips.txt` | Reddit r/berkeley wiki | Major declaration, course sequencing, registration |
| `cs_internship_recruiting_tips.txt` | Reddit + student blogs | Recruiting timeline, referrals, salary context |
| `campus_dining_reviews.txt` | Reddit + Yelp | Dining hall comparison, hours, wait times |
| `housing_off_campus_reviews.txt` | Reddit r/berkeley | Neighborhoods, lease tips, rent control |
| `mental_health_resources.txt` | Reddit r/berkeley | CAPS wait times, peer support, DSP accommodations |

---

## Chunking Strategy

I split on paragraph boundaries (double newlines) first so each chunk maps to one natural unit — one review, one paragraph, one piece of advice. Chunks are capped at 600 characters (~120-150 words) with 100-character overlap. When a paragraph is longer than 600 characters, it falls back to fixed-size splitting so facts that span paragraph boundaries don't get silently cut in half.

600 characters is big enough to hold a complete opinion or factual claim, small enough to keep retrieval precise. Below 200 characters chunks are just meaningless fragments. Above 1,000 characters and one chunk covers too many topics, which dilutes retrieval accuracy.

One thing that diverged from my original design: numbered lists (like the CS186 topic lists) often don't end with a double newline, so they weren't being split cleanly on paragraphs alone. I added the fixed-size fallback to handle those cases.

### Sample Chunks

Chunk 1 — `prof_hilfinger_cs61b.txt`
```
Source: Rate My Professors - Paul Hilfinger, CS61B Data Structures, UC Berkeley

Review 1 (Quality: 4/5, Difficulty: 5/5):
Hilfinger is a legend. CS61B with him is brutal but you come out knowing data structures cold.
His exams are notoriously hard — expect exam averages around 50-60%. The curve is generous though.
He does not hold your hand. Office hours are crowded and unhelpful if you haven't already tried
the problem yourself for an hour.
```

Chunk 2 — `cs186_databases_reviews.txt`
```
Reddit post (r/berkeley, 2023):
"What to know going into CS186: 1) Disk I/O cost is the primary metric in database systems —
know why we care about number of page reads/writes, not just CPU time. 2) B+ trees: the order (d)
determines max/min capacity, not 'page size'. 3) Two-phase locking: understand shared vs exclusive
locks, the upgrade protocol, and when deadlock can occur. 4) ARIES: the three phases are
Analysis, Redo, Undo..."
```

Chunk 3 — `campus_dining_reviews.txt`
```
CLARK KERR (Clark Kerr campus, 10 min walk from main campus):
Underrated and chronically underused by people who don't live near it. Less crowded than
Crossroads. The pasta station is consistently good. Stir-fry made to order on Thursdays.
Quiet enough to study while eating.
```

Chunk 4 — `cs_internship_recruiting_tips.txt`
```
Specific timelines:
- Jane Street / Two Sigma / Citadel (finance/quant): Applications open in July-August.
  These companies move extremely fast. If you miss the window, that's it.
- Amazon: Applications open September, OAs go out October-November, offers by December.
- Google / Meta: Applications October-November, OAs/interviews December-January.
- Microsoft / Apple: Applications October-December, interviews January-March.
```

Chunk 5 — `mental_health_resources.txt`
```
CAPS (Counseling and Psychological Services):
The official service. Free for enrolled students. The problem: 2-4 week wait time for the first
appointment in most semesters. If you're in crisis, do not use CAPS as your emergency option —
call the 24/7 crisis line instead (Tang Center: 510-642-9494). Pro tip: Call at 8am when
appointments are released — same-day slots disappear within minutes.
```

---

## Embedding Model

I used `all-MiniLM-L6-v2` via `sentence-transformers`. It runs locally with no API key, produces 384-dimensional vectors, and works well for short-to-medium English text. It was trained on Q&A pairs, which makes it a natural fit for matching student questions to review-style answers.

Tradeoffs I'd consider for production:

- Context length: this model caps at 256 tokens (~512 chars) per chunk. For longer documents, `nomic-embed-text` (8,192 tokens) or OpenAI's `text-embedding-3-large` would be needed.
- Cost: local embedding is free. OpenAI's `text-embedding-3-small` is ~$0.02/1M tokens — fine here, meaningful at scale.
- Latency: local CPU inference is ~20ms per query. For high traffic, GPU-hosted APIs (Cohere, VoyageAI) would cut this significantly.
- Domain accuracy: general models can misread domain-specific terms (e.g. "ARIES" in a database recovery context). A fine-tuned model would improve this.

---

## Retrieval Test Results

### "What is the difference between CS61B with Hilfinger vs. Hug?"

Top chunks retrieved:
1. `prof_hug_cs61b.txt` — "Honest comparison: Hug vs Hilfinger. Hug is better for most students. Hilfinger is better if you want maximum rigor..." *(distance: 0.15)*
2. `prof_hilfinger_cs61b.txt` — "Not for the faint of heart. His communication style is dense and assumes background most freshmen don't have..." *(distance: 0.18)*
3. `prof_hug_cs61b.txt` — "Josh Hug is hands down one of the best professors I've had at Berkeley. Compared to Hilfinger, Hug's version is more scaffolded..." *(distance: 0.21)*

The query explicitly named both professors, so ChromaDB returned chunks that directly compare them. Semantic similarity correctly ranked comparison-focused reviews above general CS61B content.

### "What exam topics should I know for CS186?"

Top chunks retrieved:
1. `cs186_databases_reviews.txt` — "SQL: SELECT, JOIN, GROUP BY, HAVING, subqueries, window functions. Relational algebra: selection, projection, join..." *(distance: 0.19)*
2. `cs186_databases_reviews.txt` — "B+ trees are heavily tested — practice 20-30 tree modification problems by hand before the midterm." *(distance: 0.23)*

The retriever correctly identified CS186-specific chunks and captured the "what to study" intent even though the documents phrase it as "key exam topics" rather than "exam topics to know."

### "How long does CAPS take for a mental health appointment?"

Top chunks retrieved:
1. `mental_health_resources.txt` — "The problem: 2-4 week wait time for the first appointment in most semesters..." *(distance: 0.22)*
2. `mental_health_resources.txt` — "Pro tip: Call at 8am when appointments are released — same-day slots disappear within minutes." *(distance: 0.29)*

"How long" mapped semantically to "wait time" and "weeks." The retriever surfaced both the general wait time and the same-day workaround from adjacent paragraphs.

---

## Grounded Generation

Grounding is enforced through the system prompt in `query.py`:

```
You are the Unofficial Berkeley CS Guide...

CRITICAL RULES:
1. Answer ONLY using information explicitly stated in the CONTEXT DOCUMENTS provided below.
2. Do NOT use your general training knowledge to fill in gaps, make inferences, or add
   information not present in the documents.
3. If the provided documents do not contain enough information to answer the question,
   respond with: "I don't have enough information in my documents to answer that question reliably."
4. You MUST cite your sources. At the end of every response, list the source documents you
   drew from, formatted as "Sources: [filename1, filename2]".
```

Retrieved chunks are labeled "CONTEXT DOCUMENTS" in the user prompt and the instruction says to answer using only that context. Source citation is required programmatically in rule 4 — I didn't want to rely on the model doing it on its own.

---

## Example Responses

Question: What is the difference between taking CS61B with Hilfinger vs. Hug?

> Based on student reviews, the key difference is the level of scaffolding and exam difficulty. Hilfinger's version is harder — exam averages fall around 50-60%, specs are described as "famously cryptic," and he provides little guidance. Hug's version is more student-friendly — exam averages are 60-70%, specs are "3x longer and with examples," and he personally responds to Ed posts. Students consistently recommend Hug for most learners, and Hilfinger only for those who want maximum rigor with prior experience.
>
> Sources: [prof_hilfinger_cs61b.txt, prof_hug_cs61b.txt]

Question: What dining hall is least crowded for lunch?

> According to student reviews, Clark Kerr and Foothill are the least crowded dining halls during lunch. Clark Kerr is described as "chronically underused" and Foothill as "the best-kept secret on campus — never crowded." Crossroads is flagged as the most crowded, with wait times of 10-15 minutes even outside peak hours and much worse during the 12pm-1pm rush.
>
> Sources: [campus_dining_reviews.txt]

Question: What is the best restaurant in San Francisco for a date? (out of scope)

> I don't have enough information in my documents to answer that question reliably. My documents cover UC Berkeley CS courses, campus dining, off-campus housing near Berkeley, internship recruiting, and campus mental health resources. I don't have any information about San Francisco restaurants.

---

## Web Interface

Built with Gradio `gr.Blocks`. Input is a free-text question box with Enter-to-submit. Output has three panels: the answer, a bullet list of source filenames, and an expandable accordion showing the top 5 retrieved chunks with cosine distance scores so you can see exactly what the system found.

Sample interaction:

```
User: What are the Pintos projects in CS162?

Answer:
According to student reviews, CS162 includes four Pintos projects, each building on the previous:
1. Threads — implementing a basic scheduler
2. User programs — supporting user-level programs
3. Virtual memory — implementing demand paging
4. File system — implementing a full file system

Each project is done in groups of 4. Students report 60-100 hours of group work per project.
The file system project is "the hardest one by far — most groups don't fully pass it."
Start each project the day it's released.

Sources: [cs162_os_reviews.txt]

Retrieved From:
• cs162_os_reviews.txt
```

---

## Evaluation Report

### Q1: What is the difference between taking CS61B with Hilfinger vs. Hug?

Expected: Hug is more scaffolded with clearer specs and higher exam averages (60-70%). Hilfinger has cryptic specs, lower averages (~50-60%), and less guidance. Most students recommend Hug.

System response:
> 1. Project guidance: Hug provides more starter code and step-by-step spec guidance; Hilfinger's specs are more cryptic.
> 2. Exam difficulty: Hug's averages are 60-70% vs. 50-60% for Hilfinger.
> 3. Office hours: Hilfinger requires showing your work before getting help; Hug uses a queue system.
> 4. Grading: Hug's grading is more forgiving.
> 5. Scaffolding: Hug better for most students; Hilfinger better for those wanting maximum rigor.
> Sources: prof_hug_cs61b.txt, prof_hilfinger_cs61b.txt

✅ All key comparison points present, both sources cited.

---

### Q2: What are the main topics I should study for CS186?

Expected: SQL, relational algebra, B+ trees, buffer management (LRU/MRU/CLOCK), two-phase locking, ARIES.

System response:
> - SQL: SELECT, JOIN (inner/outer/cross), GROUP BY, HAVING, subqueries, window functions
> - Relational algebra: selection, projection, join, union, difference
> - B+ tree indexing: search, insertion, deletion, order d
> - Buffer management: replacement policies (LRU, MRU, CLOCK)
> - Query optimization: cost estimation, join ordering, access paths
> - Transaction management: ACID properties, two-phase locking, deadlock detection
> - Recovery: ARIES protocol, write-ahead logging (WAL)
> Sources: cs186_databases_reviews.txt

✅ Complete topic list, accurately reproduced.

---

### Q3: Which dining hall has the shortest wait times?

Expected: Clark Kerr and Foothill. Crossroads worst during 12-1pm. Cafe 3 dinner-only.

System response:
> Clark Kerr is described as "less crowded" than Crossroads, but specific wait times are not provided. Crossroads has wait times of 10-15 minutes during lunch and 5-10 minutes outside peak hours. It can be inferred that the shortest wait times occur at Crossroads outside of peak hours.
> Sources: campus_dining_reviews.txt

⚠️ Partial — mentioned Clark Kerr as less crowded but concluded "Crossroads outside peak hours" instead of clearly naming Clark Kerr and Foothill as least crowded.

The retriever surfaced the Crossroads hours chunk first (cosine distance 0.35) because it had explicit minute-estimates. The Clark Kerr "chronically underused" chunk was qualitative and ranked lower. The LLM over-indexed on the quantitative data and missed the qualitative comparison. The most useful answer was split across two lower-ranked chunks. Increasing top-k from 5 to 7 might surface the Clark Kerr chunk, or storing each dining hall in a separate document would make retrieval more targeted. A re-ranking step (cross-encoder) after initial retrieval would also help.

---

### Q4: When should I start applying for Amazon software engineering internships?

Expected: September. OAs October-November, offers by December.

System response:
> Amazon opens applications for software engineering internships in September for the following summer.
> Sources: cs_internship_recruiting_tips.txt

✅ Correct, concise, on-topic.

---

### Q5: How long is the wait for a first CAPS appointment?

Expected: 2-4 weeks. Same-day: call at 8am. Emergency: Tang crisis line (510-642-9494).

System response:
> The typical wait for a first CAPS appointment is 2-4 weeks in most semesters.
> Sources: mental_health_resources.txt

⚠️ Partial — correctly stated the 2-4 week wait but skipped the "call at 8am for same-day slots" tip and the crisis line number (510-642-9494), both of which were in the retrieved chunks. The LLM had the information but chose not to include it.

---

## AI Tools I Used

For `embed.py` I gave Claude my planning doc's retrieval section and asked it to implement a Python script that initializes ChromaDB, embeds chunks from `ingest.py` using sentence-transformers, and stores them with source metadata. The generated code worked but I made two changes: removed a custom embedding function it added (I wanted explicit control over embedding), and added batch upsert in groups of 100 to prevent memory issues.

For `app.py` I gave Claude the Gradio skeleton from the project spec and my `query.py` function signature. It built the `gr.Blocks()` app correctly but only included two output fields. I added the third "Retrieved Context Chunks" accordion myself — users should be able to see what the system found, not just the final answer.
