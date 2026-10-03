"""Autonomous arXiv Research Agent (CS480 Lab 2).

Run locally with:  python app.py

A smolagents CodeAgent with three custom tools: it searches arXiv, reads paper
PDFs, summarizes them and formats BibTeX. The guardrails live in two places --
inside the tools (see tools.py) and around the agent loop below. The UI streams
the agent's ReAct steps live, exposes the guardrail settings in a sidebar, and
exports the papers it found as a .bib file.
"""

import os
import re
import tempfile

import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel
from smolagents.memory import ActionStep

import tools
from tools import format_bibtex, read_paper_pdf, search_arxiv

load_dotenv(".env")  # reads HF_TOKEN (and optional settings) from a local .env file

DEFAULT_STEPS = 6        # guardrail: default cap on Thought -> Code -> Observation loops
MAX_PROMPT_CHARS = 600   # guardrail: reject very long prompts before they cost tokens
TOOL_NAMES = ("search_arxiv", "read_paper_pdf", "format_bibtex")

AGENT_RULES = """You are a research assistant for arXiv papers. Follow these rules:
1. Only report papers, authors, years, and URLs that a tool returned. Never invent or
   "recall" a citation. If search_arxiv returns an empty list or an 'error' entry, say so.
2. If a tool result starts with 'GUARDRAIL:', the lookup failed. Tell the user what failed
   in plain words and stop; do not fill the gap from memory.
3. For BibTeX, call format_bibtex once per paper with the values from search_arxiv.
4. For a methodology or content summary, call read_paper_pdf and summarize only what the
   returned text says.
5. Use as few steps as possible. Call search_arxiv once per topic: pass only the topic as the
   query and set recent=True when the user wants recent papers. The tool already retries with
   looser searches by itself, so do not repeat a search with reworded queries more than once.
   If the results are still off-topic, return what was found and say it may not be relevant.
6. If the request is not about research papers, say that you only handle arXiv research
   tasks instead of using a tool.
Return the final answer as Markdown, with BibTeX inside a code block."""

# Model-provider failures worth explaining in plain words: (substrings, explanation).
ERROR_HINTS = (
    (("401", "Unauthorized"), "The API token was rejected. Check HF_TOKEN in your .env file."),
    (("402", "Payment Required", "exceeded"),
     "The free inference quota is used up. Wait for it to reset, or set "
     "OPENROUTER_API_KEY and MODEL_PROVIDER=openrouter in .env."),
    (("429",), "The model provider is rate limiting requests. Wait a minute and try again."),
)


def build_model():
    """Pick the LLM backend: Hugging Face by default, OpenRouter as a fallback."""
    hf_token = os.environ.get("HF_TOKEN")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if openrouter_key and (os.environ.get("MODEL_PROVIDER", "").lower() == "openrouter" or not hf_token):
        from smolagents import OpenAIServerModel
        model_id = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
        return OpenAIServerModel(model_id=model_id, api_key=openrouter_key,
                                 api_base="https://openrouter.ai/api/v1"), f"{model_id} (OpenRouter)"

    if not hf_token:
        raise SystemExit("No API key found. Put HF_TOKEN=hf_... in the .env file in this "
                         "folder (or set OPENROUTER_API_KEY) and run the app again.")
    model_id = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
    return InferenceClientModel(model_id=model_id, token=hf_token), f"{model_id} (Hugging Face)"


def build_agent(model):
    """Create the CodeAgent, returning (agent, whether AGENT_RULES went into it)."""
    kwargs = dict(tools=[search_arxiv, read_paper_pdf, format_bibtex], model=model,
                  add_base_tools=False, max_steps=DEFAULT_STEPS)
    try:
        return CodeAgent(**kwargs, instructions=AGENT_RULES), True
    except TypeError:  # older smolagents: rules are prepended to each task instead
        return CodeAgent(**kwargs), False


model, MODEL_LABEL = build_model()
agent, RULES_BUILT_IN = build_agent(model)

# A Hugging Face ZeroGPU Space only starts if one @spaces.GPU function exists.
# Locally there is no SPACE_ID, so this is skipped and `spaces` is not needed.
if os.environ.get("SPACE_ID"):
    try:
        import spaces

        @spaces.GPU
        def _zero_gpu_startup_check():
            pass
    except ImportError:
        pass


# --- Reading the agent's memory ----------------------------------------------
def _steps(skip_max_steps_error: bool = False) -> list:
    """The agent's steps from its last run.

    The library appends an extra step carrying the "Reached max steps" error;
    it is real history but not a ReAct loop, so the counter skips it.
    """
    steps = [s for s in agent.memory.steps if hasattr(s, "step_number")]
    if not skip_max_steps_error:
        return steps
    return [s for s in steps if "max steps" not in str(getattr(s, "error", "")).lower()]


def _hit_step_limit() -> bool:
    return len(_steps()) != len(_steps(skip_max_steps_error=True))


def _run_tokens() -> tuple:
    """Total (input, output) tokens across the steps of the last run."""
    tin = tout = 0
    for step in _steps():
        usage = getattr(step, "token_usage", None)
        if usage is not None:
            tin += getattr(usage, "input_tokens", 0) or 0
            tout += getattr(usage, "output_tokens", 0) or 0
    return tin, tout


def _trace_markdown(max_steps: int) -> str:
    """Render the last run for the UI: code written, tool output, steps, tokens."""
    steps = _steps()
    if not steps:
        return "_Waiting for the agent..._"

    counts = dict.fromkeys(TOOL_NAMES, 0)
    blocks = []
    for step in steps:
        code = (getattr(step, "code_action", None) or "").strip()
        observation = (getattr(step, "observations", None) or "").strip()
        error = getattr(step, "error", None)
        for name in TOOL_NAMES:  # the agent calls tools by writing Python, so count call sites
            counts[name] += code.count(name + "(")

        block = f"**Step {step.step_number}**\n"
        if code:
            block += f"\n```python\n{code}\n```\n"
        if observation:
            block += "\nObservation:\n\n```text\n" + (
                observation[:1200] + "\n... (trimmed)" if len(observation) > 1200 else observation) + "\n```\n"
        if error:
            block += f"\nError: `{str(error)[:300]}`\n"
        blocks.append(block)

    tin, tout = _run_tokens()
    calls = ", ".join(f"{name} x{n}" for name, n in counts.items() if n) or "none"
    return (f"**ReAct steps:** {len(_steps(skip_max_steps_error=True))} of {max_steps} allowed "
            f"&nbsp;|&nbsp; **Tool calls:** {calls} &nbsp;|&nbsp; "
            f"**Tokens:** {tin:,} in / {tout:,} out\n\n" + "\n---\n".join(blocks))


# --- Guardrails that run after the agent finishes ----------------------------
def _unverified_citations(answer: str, prompt: str) -> list:
    """Return the arXiv IDs and BibTeX titles in the answer that no tool returned
    during this run. IDs the user typed in the prompt are allowed, since quoting
    the request back is not a fabricated citation."""
    allowed = tools.RUN_LOG["ids"] | tools.find_arxiv_ids(prompt)
    problems = [f"arXiv ID {bad}" for bad in sorted(tools.find_arxiv_ids(answer) - allowed)]
    for title in re.findall(r"^\s*title\s*=\s*[{\"](.+?)[}\"],?\s*$", answer, flags=re.M | re.I):
        if tools.normalize_title(title) not in tools.RUN_LOG["titles"]:
            problems.append(f'title "{title.strip()}"')
    return problems


def _dedup_papers() -> list:
    """The papers the tools returned this run, de-duplicated by arXiv ID."""
    seen, unique = set(), []
    for paper in tools.RUN_LOG["papers"]:
        key = tools.normalize_id(paper["arxiv_id"])
        if key not in seen:
            seen.add(key)
            unique.append(paper)
    return unique


def _verified_papers_markdown() -> str:
    """List the verified papers, shown when an answer is blocked."""
    papers = _dedup_papers()
    if not papers:
        return "The search tool did not return any papers in this run."
    lines = [f"- **{p['title']}** ({p['year']}), {p['authors']}. {p['url']}" for p in papers]
    return "Papers the search tool actually returned in this run:\n\n" + "\n".join(lines)


def _finalize(answer: str, prompt: str, max_steps: int) -> str:
    """Apply the citation check, wrap bare BibTeX, and flag a hit step limit."""
    problems = _unverified_citations(answer, prompt)
    if problems:  # the model cited something no tool returned, so the answer is discarded
        return ("**Guardrail: answer blocked.** The agent's draft cited papers that none of its "
                "tools returned (" + "; ".join(problems[:6]) + "), so it was discarded as "
                "unverified.\n\n" + _verified_papers_markdown())
    if "```" not in answer:  # wrap bare BibTeX so Markdown keeps its line breaks
        answer = re.sub(r"((?:@\w+\{.*?\n\}\s*)+)",
                        lambda m: "```bibtex\n" + m.group(1).strip() + "\n```\n", answer, flags=re.S)
    if _hit_step_limit():
        answer = (f"*Note: the agent reached its {max_steps}-step limit, so this answer may be "
                  "incomplete. Every citation in it was checked against the tool results.*\n\n") + answer
    return answer


def _bib_file() -> str | None:
    """Write this run's verified papers to a .bib file; return its path (or None)."""
    papers = _dedup_papers()
    if not papers:
        return None
    entries = [format_bibtex(p["title"], p["authors"], p["year"], p["url"]) for p in papers]
    path = tempfile.NamedTemporaryFile(prefix="arxiv_", suffix=".bib", delete=False, mode="w",
                                       encoding="utf-8").name
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(e for e in entries if not e.startswith("GUARDRAIL")))
    return path


MEMORY_PAPERS = 12  # cap on how many earlier papers are carried into a follow-up


def _memory_context(mem: dict) -> str:
    """Compact recap of the papers found earlier this conversation.

    Short-term memory is done at the task level, not by replaying the agent's
    full transcript: carrying a short numbered list of earlier papers is far
    cheaper in tokens and is enough for follow-ups like "summarize the second
    one" or "format them as BibTeX" to resolve.
    """
    papers = mem.get("papers", [])[-MEMORY_PAPERS:]
    if not papers:
        return ""
    lines = [f'{i}. "{p["title"]}" -- {p["authors"]} ({p["year"]}), '
             f'arXiv:{p["arxiv_id"]}, {p["url"]}' for i, p in enumerate(papers, 1)]
    return ("Context -- papers already found earlier in this conversation. Reuse these "
            "directly for follow-up questions (e.g. 'the second one', 'all of them'); only "
            "call search_arxiv again if the user clearly asks for different or new papers:\n"
            + "\n".join(lines) + "\n\n")


# --- The run loop (a generator, so the UI streams each step live) ------------
def agent_chat(user_prompt: str, max_steps: int, max_results: int, prefer_recent: bool,
               mem: dict, stats: dict):
    """Run the agent on one prompt, yielding the six UI outputs (see `outputs` below)."""
    mem = mem or {"papers": []}

    def out(answer, download=gr.update()):
        return answer, _trace_markdown(max_steps), download, _stats_markdown(stats), mem, stats

    prompt = (user_prompt or "").strip()
    if not prompt:
        yield out("Please type a research request first.", gr.update(visible=False))
        return
    if len(prompt) > MAX_PROMPT_CHARS:
        yield out(f"**Guardrail:** that prompt is {len(prompt)} characters long. "
                  f"Please keep requests under {MAX_PROMPT_CHARS} characters.", gr.update(visible=False))
        return

    # Build the task: earlier-paper context (memory) + this prompt + sidebar preferences.
    task = (_memory_context(mem) + f"User request: {prompt}\n\n"
            f"[Preferences: return up to {max_results} papers; "
            f"prefer recent papers = {'yes' if prefer_recent else 'no'}.]")
    if not RULES_BUILT_IN:
        task = f"{AGENT_RULES}\n\n{task}"

    # Fresh run log, then re-seed it with remembered papers so a follow-up that cites
    # one of them passes the citation-verification guardrail.
    tools.reset_run_log()
    if mem["papers"]:
        tools._remember(mem["papers"])

    yield out("_Working... the agent's steps will appear below as it runs._", gr.update(visible=False))

    final_answer = ""
    try:
        # stream=True yields each step as it completes, so the trace panel fills live.
        # reset=True keeps per-turn memory small; conversation context is the recap above.
        for event in agent.run(task, stream=True, reset=True, max_steps=int(max_steps)):
            if isinstance(event, ActionStep):
                yield out("_Working..._", gr.update(visible=False))
            elif not hasattr(event, "step_number"):  # FinalAnswerStep (or similar) at the end
                final_answer = str(getattr(event, "output", None) or getattr(event, "final_answer", "") or "")
    except Exception as exc:
        message = str(exc)
        hint = next((explanation for needles, explanation in ERROR_HINTS
                     if any(needle.lower() in message.lower() for needle in needles)),
                    "The agent stopped because of an unexpected error.")
        yield out(f"**The agent could not finish.** {hint}\n\n`{type(exc).__name__}: {message[:400]}`",
                  gr.update(visible=False))
        return

    answer = _finalize(final_answer, prompt, int(max_steps))

    # Remember every paper seen so far (old seeds + anything found this turn) for follow-ups.
    mem = {"papers": _dedup_papers()}
    tin, tout = _run_tokens()
    stats = {"runs": stats["runs"] + 1, "tokens_in": stats["tokens_in"] + tin,
             "tokens_out": stats["tokens_out"] + tout}
    bib = _bib_file()
    yield out(answer, gr.update(value=bib, visible=bib is not None))


def _stats_markdown(stats: dict) -> str:
    total = stats["tokens_in"] + stats["tokens_out"]
    return (f"**Runs this session:** {stats['runs']}  \n"
            f"**Total tokens:** {total:,}  \n"
            f"<span style='color:var(--body-text-color-subdued)'>"
            f"{stats['tokens_in']:,} in / {stats['tokens_out']:,} out</span>")


EXAMPLES = [
    "Find 2 recent papers on Neural Stochastic Differential Equations and format them as BibTeX.",
    "Read the paper with arXiv ID 1706.03762 and summarize its methodology in 5 bullet points.",
    "Summarize the methodology of the paper with arXiv ID 9999.99999.",
    "Find 3 papers about zxqvjw plorfnagh quibblestrum and format them as BibTeX.",
]

# Light theme with blue accents. Soft is a light base; the force-light JS below
# stops a dark-mode browser from overriding it.
THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

# One-time redirect that pins the page to light mode regardless of OS/browser setting.
FORCE_LIGHT = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light'); window.location.replace(u.href);
  }
}"""

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto !important;
    background: #f5f8ff !important;}
/* Animated "agentic" glow frame around the whole viewport. pointer-events:none
   keeps it purely decorative so it never blocks clicks. */
body::after {
    content: ""; position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    border: 1.5px solid rgba(37,99,235,.55); border-radius: 6px;
    animation: agentic-glow 3.2s ease-in-out infinite;
}
@keyframes agentic-glow {
    0%,100% {box-shadow: inset 0 0 12px rgba(37,99,235,.30), inset 0 0 28px rgba(56,189,248,.14);}
    50%     {box-shadow: inset 0 0 26px rgba(37,99,235,.55), inset 0 0 60px rgba(56,189,248,.30);}
}
#hero {background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
       color: #fff; padding: 22px 26px; border-radius: 14px; margin-bottom: 14px;
       box-shadow: 0 6px 24px rgba(37,99,235,.28);}
#hero h1 {margin: 0 0 4px; font-size: 1.6rem; font-weight: 700;}
#hero p {margin: 0; opacity: .92; font-size: .95rem;}
#hero code {background: rgba(255,255,255,.20); color: #fff; padding: 1px 6px; border-radius: 5px;}
.sb-card {background: #eef4ff; border: 1px solid #c7d8f5;
          border-radius: 10px; padding: 10px 12px; font-size: .85rem; line-height: 1.5;}
footer {display: none !important;}
"""

with gr.Blocks(title="Autonomous arXiv Research Agent", theme=THEME, css=CSS) as demo:
    stats = gr.State({"runs": 0, "tokens_in": 0, "tokens_out": 0})
    memory = gr.State({"papers": []})  # short-term memory of papers seen this conversation

    gr.HTML(f"""<div id="hero">
      <h1>Autonomous arXiv Research Agent</h1>
      <p>Searches arXiv, reads papers and formats citations — a smolagents ReAct agent
      running <code>{MODEL_LABEL}</code>. Remembers the current conversation, so you can ask
      follow-ups.</p></div>""")

    with gr.Sidebar(label="Settings", open=True):
        gr.Markdown("### Settings")
        steps_slider = gr.Slider(2, 8, value=DEFAULT_STEPS, step=1, label="Max ReAct steps",
                                 info="Hard cap on the agent's reasoning loops.")
        results_slider = gr.Slider(1, 5, value=3, step=1, label="Max papers",
                                   info="How many papers a search may return.")
        recent_toggle = gr.Checkbox(value=False, label="Prefer recent papers",
                                    info="Order results newest-first instead of by relevance.")
        gr.Markdown("### Session")
        stats_md = gr.Markdown(_stats_markdown({"runs": 0, "tokens_in": 0, "tokens_out": 0}))
        gr.Markdown("### Guardrails")
        gr.Markdown(
            "<div class='sb-card'>Citation verification · honest tool failures · "
            "input validation · abstract &amp; PDF caps · visible provider errors</div>")

    with gr.Row():
        prompt_box = gr.Textbox(lines=2, label="Your request", scale=5, autofocus=True,
                                placeholder="E.g., Find 2 recent papers on Agentic AI and format them as BibTeX.")
    with gr.Row():
        submit = gr.Button("Search", variant="primary", scale=3)
        clear = gr.Button("New conversation", scale=1)
    gr.Markdown("<span style='font-size:.85rem;color:#475569'>Ask a follow-up and the agent "
                "reuses what it already found. \"New conversation\" clears that memory.</span>")

    answer_box = gr.Markdown(label="Agent output", value="_Results will appear here._")
    download_btn = gr.DownloadButton("Download citations (.bib)", visible=False, size="sm")
    with gr.Accordion("What the agent did (steps, tool calls, tokens)", open=False):
        trace_box = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=prompt_box, label="Try one of these")

    inputs = [prompt_box, steps_slider, results_slider, recent_toggle, memory, stats]
    outputs = [answer_box, trace_box, download_btn, stats_md, memory, stats]
    submit.click(agent_chat, inputs=inputs, outputs=outputs)
    prompt_box.submit(agent_chat, inputs=inputs, outputs=outputs)
    # "New conversation" clears the input, the panels, and the short-term memory.
    clear.click(lambda: ("", "_Results will appear here._", "", gr.update(visible=False), {"papers": []}),
                outputs=[prompt_box, answer_box, trace_box, download_btn, memory])

    demo.load(None, js=FORCE_LIGHT)  # pin to light mode on page load (supported idiom)

if __name__ == "__main__":
    demo.launch()
