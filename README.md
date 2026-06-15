# The Unofficial Berkeley CS Guide 🐻

A RAG (Retrieval-Augmented Generation) system that makes student-generated knowledge about UC Berkeley CS courses, campus life, housing, and internships searchable and answerable.

---

## Domain and Document Sources

**Domain:** UC Berkeley CS — Student-Generated Course & Campus Knowledge

The CS department publishes official course catalogs and syllabi, but not the practical, experience-based knowledge that determines whether students succeed. This includes: which professor's exam averages 50% vs. 70%, which dining hall has a 45-minute line, how off-campus housing leases actually work, and when to start applying for internships. This knowledge circulates on Reddit, Rate My Professors, and Discord — but is fragmented and inaccessible without knowing where to look.

### Document Sources (14 documents)

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

## Chunking Strategy and Reasoning

**Method:** Paragraph-aware splitting with fixed-size fallback
**Chunk size:** 600 characters (~120–150 words)
**Overlap:** 100 characters (~20 words)

The documents are a mix of short review-style text (2-4 sentences per opinion) and longer informational sections (exam topic lists, neighborhood guides). A pure fixed-size character split would often slice a single review or sentence mid-thought, destroying its meaning. Splitting on double newlines first keeps each review or paragraph together as a natural unit of meaning.

When a paragraph exceeds 600 characters, it's split with 100-character overlap. The overlap ensures that facts spanning a paragraph boundary (e.g., "exam average is 55%" at the end of one paragraph, "the curve brings it to B+" at the start of the next) appear in at least one chunk together.

**Why 600 characters?** Large enough to contain a complete opinion or factual statement; small enough to match specific queries. Chunks below 200 characters are often fragments without standalone meaning. Chunks above 1,000 characters cover too many topics and dilute retrieval precision.

### Sample Chunks (5 labeled examples)

**Chunk 1** — `prof_hilfinger_cs61b.txt` (chunk_index: 0)
```
Source: Rate My Professors - Paul Hilfinger, CS61B Data Structures, UC Berkeley
URL: https://www.ratemyprofessors.com/professor/hilfinger-cs61b

Review 1 (Quality: 4/5, Difficulty: 5/5):
Hilfinger is a legend. CS61B with him is brutal but you come out knowing data structures cold. His exams are notoriously hard — expect exam averages around 50-60%. The curve is generous though. He does not hold your hand. Office hours are crowded and unhelpful if you haven't already tried the problem yourself for an hour.
```

**Chunk 2** — `cs186_databases_reviews.txt` (chunk_index: 3)
```
Reddit post (r/berkeley, 2023):
"What to know going into CS186: 1) Disk I/O cost is the primary metric in database systems — know why we care about number of page reads/writes, not just CPU time. 2) B+ trees: the order (d) determines max/min capacity, not 'page size'. 3) Two-phase locking: understand shared vs exclusive locks, the upgrade protocol, and when deadlock can occur. 4) ARIES: the three phases are Analysis, Redo, Undo..."
```

**Chunk 3** — `campus_dining_reviews.txt` (chunk_index: 2)
```
CLARK KERR (Clark Kerr campus, 10 min walk from main campus):
Underrated and chronically underused by people who don't live near it. Less crowded than Crossroads. The pasta station is consistently good. Stir-fry made to order on Thursdays. Quiet enough to study while eating. The dessert rotation is better than other halls — actual pies and tarts, not just soft-serve.
```

**Chunk 4** — `cs_internship_recruiting_tips.txt` (chunk_index: 1)
```
Specific timelines:
- Jane Street / Two Sigma / Citadel (finance/quant): Applications open in July-August. These companies move extremely fast. If you miss the window, that's it.
- Amazon: Applications open September, OAs (online assessments) go out October-November, offers by December.
- Google / Meta: Applications October-November, OAs/interviews December-January.
- Microsoft / Apple: Applications October-December, interviews January-March.
```

**Chunk 5** — `mental_health_resources.txt` (chunk_index: 0)
```
CAPS (Counseling and Psychological Services):
The official service. Free for enrolled students. The problem: 2-4 week wait time for the first appointment in most semesters. If you're in crisis, do not use CAPS as your emergency option — call the 24/7 crisis line instead (Tang Center: 510-642-9494). For non-emergency ongoing therapy, CAPS is fine but slow to start. Pro tip: Call at 8am when appointments are released — same-day slots disappear within minutes.
```

---

## Embedding Model

**Model:** `all-MiniLM-L6-v2` via `sentence-transformers`

This model runs locally with no API key and no rate limits. It produces 384-dimensional vectors and performs well on English semantic similarity for short-to-medium text. It was trained on diverse corpora including Q&A pairs, making it well-suited for matching student questions to review-style answers.

**Production tradeoffs I would consider:**
- **Context length:** `all-MiniLM-L6-v2` has a 256-token (~512 character) input limit per chunk. For longer documents, `nomic-embed-text` (8,192 tokens) or OpenAI's `text-embedding-3-large` would be needed.
- **Cost:** Local embedding has zero marginal cost. OpenAI's `text-embedding-3-small` costs $0.02/1M tokens — negligible here but meaningful at scale.
- **Multilingual support:** Not needed for this English-only corpus. For multilingual content, `paraphrase-multilingual-MiniLM-L12-v2` would be appropriate.
- **Domain accuracy:** General models can misinterpret domain-specific terms (e.g., "ARIES" in database recovery context). A fine-tuned domain model would improve retrieval quality.
- **Latency:** Local CPU inference is ~20ms per query. For high-traffic production, GPU-hosted APIs (Cohere, VoyageAI) would reduce latency significantly.

---

## Retrieval Test Results

### Query 1: "What is the difference between CS61B with Hilfinger vs. Hug?"

**Top Retrieved Chunks:**
1. `prof_hug_cs61b.txt` — "Honest comparison: Hug vs Hilfinger. Hug is better for most students. Hilfinger is better if you want maximum rigor and already have programming experience. Hug's grading is more forgiving..." *(distance: 0.15)*
2. `prof_hilfinger_cs61b.txt` — "Not for the faint of heart. His communication style is dense and assumes background most freshmen don't have. Dropped once, retook with Hug (easier) the second time..." *(distance: 0.18)*
3. `prof_hug_cs61b.txt` — "Josh Hug is hands down one of the best professors I've had at Berkeley. Compared to Hilfinger, Hug's version is more scaffolded..." *(distance: 0.21)*

**Why these chunks are relevant:** The query asks for a comparison between two specific professors. ChromaDB returns chunks that explicitly contain both names and use comparative language ("Hug vs. Hilfinger," "better for most students"). The semantic similarity correctly identifies that comparison-focused reviews are more relevant than general CS61B content.

### Query 2: "What exam topics should I know for CS186?"

**Top Retrieved Chunks:**
1. `cs186_databases_reviews.txt` — "SQL: SELECT, JOIN (inner/outer/cross), GROUP BY, HAVING, subqueries, window functions. Relational algebra: selection, projection, join..." *(distance: 0.19)*
2. `cs186_databases_reviews.txt` — "The SQL homeworks are straightforward if you've used SQL before... B+ trees are heavily tested — practice 20-30 tree modification problems by hand before the midterm." *(distance: 0.23)*

**Why these chunks are relevant:** The query contains "exam topics" and "CS186" — the retriever correctly identifies CS186-specific chunks that enumerate study material. The semantic search captures "what to study" meaning even though the documents say "key exam topics" rather than "exam topics to know."

### Query 3: "How long does CAPS take for a mental health appointment?"

**Top Retrieved Chunks:**
1. `mental_health_resources.txt` — "The problem: 2-4 week wait time for the first appointment in most semesters. If you're in crisis, do not use CAPS as your emergency option..." *(distance: 0.22)*
2. `mental_health_resources.txt` — "Pro tip: Call at 8am when appointments are released — same-day slots disappear within minutes." *(distance: 0.29)*

**Why these chunks are relevant:** "How long" maps semantically to "wait time" and "weeks." The retriever successfully finds both the general wait time (2-4 weeks) and the same-day workaround tip from adjacent paragraphs.

---

## Grounded Generation

Grounding is enforced through the **system prompt** in `query.py`:

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

The user prompt structure also reinforces grounding — retrieved chunks are labeled as "CONTEXT DOCUMENTS" and the instruction says "answer using ONLY the information in the context documents above." Source attribution is required programmatically in the prompt (rule 4), not left to the model's discretion.

---

## Example Responses

### Example 1 — Grounded response with citation

**Question:** What is the difference between taking CS61B with Hilfinger vs. Hug?

**System Response:**
> Based on student reviews, the key difference between CS61B with Hilfinger vs. Hug is the level of scaffolding and exam difficulty. Hilfinger's version is harder — exam averages fall around 50-60%, specs are described as "famously cryptic," and he provides little guidance ("he does not hold your hand"). Office hours are crowded and unhelpful if you haven't already tried the problem yourself. Hug's version is more student-friendly — exam averages are 60-70%, specs are described as "3x longer and with examples," and he personally responds to Ed posts. Students consistently recommend Hug for most learners, and Hilfinger only for those who want maximum rigor and already have programming experience.
>
> Sources: [prof_hilfinger_cs61b.txt, prof_hug_cs61b.txt]

### Example 2 — Grounded response about dining

**Question:** What dining hall is least crowded for lunch?

**System Response:**
> According to student reviews, Clark Kerr and Foothill are the least crowded dining halls during lunch. Clark Kerr is described as "chronically underused" and Foothill as "the best-kept secret on campus — never crowded." Crossroads is specifically flagged as the most crowded, with wait times of 10-15 minutes even outside peak hours and much worse during the 12pm-1pm rush, which is described as having "the worst wait times of the week." Cafe 3 is dinner-only so it avoids the lunch rush entirely.
>
> Sources: [campus_dining_reviews.txt]

### Example 3 — Out-of-scope query showing refusal

**Question:** What is the best restaurant in San Francisco for a date?

**System Response:**
> I don't have enough information in my documents to answer that question reliably. My documents cover UC Berkeley CS courses, campus dining, off-campus housing near Berkeley, internship recruiting, and campus mental health resources. I don't have any information about San Francisco restaurants.

---

## Query Interface

**Technology:** Gradio `gr.Blocks` web UI

**Input fields:**
- `Your Question` — a free-text question input (Textbox, 3 lines)
- `Ask` button and Enter-to-submit support

**Output fields:**
- `Answer` — the LLM's grounded response (Textbox, 10 lines)
- `Retrieved From (Source Documents)` — bullet list of source filenames
- `Retrieved Context Chunks (expandable accordion)` — top 5 retrieved chunks with distance scores, for transparency

**Sample interaction transcript:**

```
User: What are the Pintos projects in CS162?

Answer:
According to student reviews, CS162 includes four Pintos projects, each building on
the previous one. These are:
1. Threads — implementing a basic scheduler
2. User programs — supporting user-level programs
3. Virtual memory — implementing demand paging and virtual memory
4. File system — implementing a full file system

Each project is done in groups of 4. Students report that each project takes
60-100 hours of group work. The file system project is described as "the hardest
one by far — most groups don't fully pass it." A design document is required before
coding for each project and is graded separately. Students are advised to start
each project the day it's released.

Sources: [cs162_os_reviews.txt]

Retrieved From:
• cs162_os_reviews.txt
```

To run the interface:
```bash
python app.py
# Then open: http://localhost:7860
```

---

## Evaluation Report

### Question 1: What is the difference between taking CS61B with Hilfinger vs. Hug?
**Expected Answer:** Hug is more scaffolded with clearer specs and higher exam averages (60-70%). Hilfinger has cryptic specs, lower averages (~50-60%), and less guidance. Most students recommend Hug.

**System Response (actual):**
> The main differences between taking CS61B with Hilfinger vs. Hug are:
> 1. Level of guidance on projects: Hug provides more starter code and step-by-step spec guidance on projects, whereas Hilfinger's specs are more cryptic.
> 2. Exam difficulty and wording: Hug's exams have averages of 60-70% compared to 50-60% for Hilfinger.
> 3. Office hours policy: Hilfinger requires students to show their work and what they've tried, while Hug's office hours use a queue system.
> 4. Grading: Hug's grading is more forgiving, more likely to adjust points if students explain their reasoning.
> 5. Scaffolding: Hug's version is more scaffolded, better suited for most students; Hilfinger's is better for those wanting maximum rigor.
> Sources: prof_hug_cs61b.txt, prof_hilfinger_cs61b.txt

**Sources Retrieved:** prof_hug_cs61b.txt, prof_hilfinger_cs61b.txt
**Accuracy:** ✅ Accurate — all key comparison points present, both source files cited.

---

### Question 2: What are the main topics I should study for CS186?
**Expected Answer:** SQL, relational algebra, B+ trees, buffer management (LRU/MRU/CLOCK), two-phase locking, and ARIES. ARIES appears on every exam.

**System Response (actual):**
> The main topics to study for the CS186 database exam include:
> - SQL: SELECT, JOIN (inner/outer/cross), GROUP BY, HAVING, subqueries, window functions
> - Relational algebra: selection, projection, join, union, difference
> - B+ tree indexing: search, insertion, deletion, order d
> - Buffer management: replacement policies (LRU, MRU, CLOCK)
> - Query optimization: cost estimation, join ordering, access paths
> - Transaction management: ACID properties, two-phase locking, deadlock detection
> - Recovery: ARIES protocol, write-ahead logging (WAL)
> Sources: cs186_databases_reviews.txt

**Sources Retrieved:** cs186_databases_reviews.txt, cs61c_arch_reviews.txt, cs61a_tips_reddit.txt
**Accuracy:** ✅ Accurate — complete topic list reproduced accurately from source documents.

---

### Question 3: Which dining hall has the shortest wait times at UC Berkeley?
**Expected Answer:** Clark Kerr and Foothill. Crossroads worst during 12-1pm. Cafe 3 dinner-only.

**System Response (actual):**
> According to the context documents, Crossroads has wait times of 10-15 minutes during lunch (11:30am-1:30pm) and 5-10 minutes outside of peak hours. Clark Kerr is described as "less crowded" than Crossroads, but specific wait times are not provided. However, it is mentioned that Clark Kerr is "quiet enough to study while eating", implying that it may have shorter wait times due to being "underrated and chronically underused." Since specific wait times are only provided for Crossroads, and it is stated that wait times are 5-10 minutes outside of peak hours, it can be inferred that the shortest wait times are 5-10 minutes, which occur at Crossroads outside of peak hours.
> Sources: campus_dining_reviews.txt

**Sources Retrieved:** campus_dining_reviews.txt
**Accuracy:** ⚠️ Partially Accurate — mentioned Clark Kerr as less crowded but concluded "Crossroads outside peak hours" instead of clearly naming Clark Kerr and Foothill as the least-crowded halls. The retriever surfaced the Crossroads hours chunk first, which skewed the response toward Crossroads.

---

### Question 4: When should I start applying for Amazon software engineering internships?
**Expected Answer:** September. OAs October-November, offers by December.

**System Response (actual):**
> According to the provided documents, Amazon opens applications for software engineering internships in September of the current year for the following summer. This means if you want an Amazon internship the summer after your sophomore year, you should be applying in September of your sophomore year.
> Sources: cs_internship_recruiting_tips.txt

**Sources Retrieved:** cs_internship_recruiting_tips.txt, cs_major_advising_tips.txt
**Accuracy:** ✅ Accurate — September opening date correctly identified and explained. Concise and on-topic.

---

### Question 5: How long is the wait for a first CAPS appointment?
**Expected Answer:** 2-4 weeks. Same-day: call at 8am. Emergency: Tang crisis line (510-642-9494).

**System Response (actual):**
> The typical wait for a first appointment at CAPS (Counseling and Psychological Services) is 2-4 weeks in most semesters.
> Sources: mental_health_resources.txt

**Sources Retrieved:** mental_health_resources.txt, cs_major_advising_tips.txt
**Accuracy:** ⚠️ Partially Accurate — correctly stated the 2-4 week wait time but did not include the "call at 8am for same-day slots" tip or the crisis line phone number (510-642-9494), both of which were in the retrieved chunks but the LLM chose not to include them.

---

### Honest Failure Case Analysis

**Question 3 (partial failure):** "Which dining hall has the shortest wait times?"

**What happened:** The retriever surfaced a chunk about Crossroads hours and wait times as the top result (cosine distance 0.35). That chunk explicitly named Crossroads wait times in minutes. Clark Kerr and Foothill were described qualitatively ("underrated," "never crowded") but without specific minute-estimates. The LLM anchored on the quantitative data (minutes at Crossroads) and reasoned that "5-10 minutes outside peak hours" was therefore the minimum, rather than inferring that qualitatively "never crowded" = shorter waits.

**Why it happened:** Chunk-level retrieval fetched the hours/waits summary chunk which centered on Crossroads data. The Clark Kerr "chronically underused" description was in a different chunk that ranked lower. The LLM was grounded correctly — it only used retrieved information — but it over-indexed on explicit quantitative data and missed the implicit qualitative comparison. This is a retrieval ordering issue: the most relevant answer ("Clark Kerr and Foothill are least crowded") was split across two lower-ranked chunks.

**Fix:** Increasing top-k from 5 to 7 might surface the Clark Kerr underuse chunk. Alternatively, storing each dining hall in separate documents rather than one combined file would produce more targeted retrieval. A re-ranking step (cross-encoder) after initial retrieval would also help surface the most answer-aligned chunks regardless of initial cosine distance ordering.

---

## Spec Reflection

**How the spec helped:** Writing the planning.md chunking strategy section before building `ingest.py` forced me to decide *why* 600 characters and 100 overlap before implementing it. This meant the implementation matched the design — when I tested chunks, they were paragraph-aligned as planned. Without the spec, I likely would have used a default TextSplitter without thinking about document structure.

**How implementation diverged:** The planning.md described using paragraph boundaries as the primary split. In practice, many documents had multi-line sections (like the numbered topic lists in CS186) that didn't end with a double newline. I added a fallback: if no double newline is found within chunk_size, fall back to fixed-size splitting. The spec didn't anticipate this edge case.

---

## AI Tool Usage

**Instance 1 — Generating `embed.py`:**
I gave Claude this planning.md's Retrieval Approach section and the Architecture diagram. I asked it to "implement a Python script that initializes ChromaDB with a persistent client, embeds chunks from ingest.py using sentence-transformers all-MiniLM-L6-v2, and stores them with source metadata." The generated code used `client.get_or_create_collection()` correctly. I changed two things: (a) removed a custom embedding function it added (ChromaDB has its own embedding function option, but I wanted explicit control), and (b) added batch upsert in groups of 100 to prevent memory issues with large collections.

**Instance 2 — Generating the Gradio interface in `app.py`:**
I gave Claude the Gradio skeleton from the project spec and my `query.py` function signature (`ask(question: str) -> dict`). I asked it to build a `gr.Blocks()` app that takes a text question and outputs answer, sources, and retrieved chunks in three separate textboxes. The generated code was correct but had only two output fields. I added the third "Retrieved Context Chunks" accordion myself because transparency about retrieved chunks is important for a demo — users should be able to see what the system found.

---

## Setup Instructions

```bash
# 1. Clone and navigate
git clone <your-fork-url>
cd week_1

# 2. Create and activate virtual environment
#    If using conda (recommended for Apple Silicon Mac):
conda create -n berkeley_rag python=3.11 -y
conda activate berkeley_rag
pip install sentence-transformers==3.4.1 "numpy<2"
pip install chromadb groq python-dotenv gradio

#    If using standard venv (Linux/Windows or Intel Mac):
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
pip install -r requirements.txt

# 3. Set up API key
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)

# 4. Build the vector store (run once — downloads ~90MB model on first run)
python embed.py

# 5. Launch the interface
python app.py
# Open: http://localhost:7860

# Optional: run evaluation
python evaluate.py
```
