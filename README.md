---
title: Arxiv Agent Lab
emoji: 🔥
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 6.27.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Agentic AI Lab
---

# Autonomous arXiv Research Agent (CS480 Lab 2)

An AI agent that searches arXiv, reads paper PDFs, summarizes them, and formats
citations as BibTeX. It extends the lab's starter agent (one search tool) with
two more tools and a set of guardrails, and runs locally.

## How it works

The agent is a `smolagents` `CodeAgent`. The LLM (Qwen 2.5 Coder 32B through
Hugging Face) writes short Python snippets that call the tools, reads what they
return, and repeats until it has an answer: Thought -> Code -> Observation.

| Tool | What it does |
| --- | --- |
| `search_arxiv(query, max_results, recent)` | Searches arXiv, exact phrase in titles and abstracts first, then looser queries only if needed. `recent=True` orders results newest first. Returns title, authors, year, arXiv ID, URL and abstract. |
| `read_paper_pdf(arxiv_id, max_chars)` | Downloads the paper's PDF, extracts the text of its first pages, and returns it so the model can summarize the methodology. |
| `format_bibtex(title, authors, year, url)` | Builds a BibTeX entry in plain Python, so citations are never written from the model's memory. |

## Guardrails

1. **Step limit.** The agent may loop at most 6 times (`MAX_STEPS`), so it cannot run forever.
2. **Citation verification.** Every paper a tool returns is recorded in a run log. Before an
   answer is shown, each arXiv ID and BibTeX title in it is checked against that log. If the
   model cites a paper no tool returned, the answer is discarded and the papers that were
   really found are listed instead. This catches the case where the model runs out of steps
   and writes citations from memory.
3. **Honest tool failures.** Tools never raise and never return fake data. A failed lookup
   returns a message starting with `GUARDRAIL:`, and the agent's instructions tell it to report
   the failure instead of answering from memory.
4. **Input validation.** arXiv IDs are checked against the real ID format before any download,
   `max_results` is clamped to 1-5, and prompts over 600 characters are rejected.
5. **Cost control.** Abstracts are trimmed to 500 characters and PDF text to 6,000 (12,000 at
   most), PDFs over 20 MB are refused, and only the first 12 pages are read.
6. **Visible failures.** Token, quota and rate-limit errors from the model provider are shown
   in the UI with a plain explanation instead of a crash.

The panel "What the agent did" under each answer shows the code the agent wrote at every
step, what the tools returned, how many steps it used, and the token count.

## Interface

- **Live steps.** The agent's ReAct loop streams into the page as it runs, so you watch each
  search, PDF read and BibTeX call happen instead of staring at a blank box.
- **Settings sidebar.** Sliders for the max ReAct steps and max papers, a "prefer recent"
  toggle, a running token/run counter for the session, and the active guardrails at a glance.
- **Export.** A "Download citations (.bib)" button writes the verified papers to a BibTeX file.

## Run it

```powershell
cd arxiv-agent-lab
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open the local URL that is printed (usually http://127.0.0.1:7860).

The app reads the API key from a `.env` file in this folder:

```
HF_TOKEN=hf_your_token_here
```

Optional fallback if the Hugging Face free quota runs out: add
`OPENROUTER_API_KEY=...` and `MODEL_PROVIDER=openrouter` to `.env`.

## Demo prompts

1. `Find 2 recent papers on Neural Stochastic Differential Equations and format them as BibTeX.`
   (search + BibTeX tool)
2. `Read the paper with arXiv ID 1706.03762 and summarize its methodology in 5 bullet points.`
   (PDF tool + summarization)
3. `Summarize the methodology of the paper with arXiv ID 9999.99999.`
   (guardrail: the paper does not exist, so the agent says so)
4. `Find 3 papers about zxqvjw plorfnagh quibblestrum and format them as BibTeX.`
   (guardrail: nothing matches, so the agent reports that instead of inventing papers)

## Files

- `app.py` - model setup, agent rules, Gradio interface
- `tools.py` - the three tools and their guardrails
- `requirements.txt` - Python dependencies
