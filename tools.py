"""Tools for the Autonomous arXiv Research Agent (CS480 Lab 2).

One guardrail rule governs every tool here: never raise, and never return
anything the model could mistake for real data. A failed lookup comes back as a
string starting with "GUARDRAIL:" (search returns ``[{"error": ...}]``), so the
agent reports the failure instead of inventing an answer.
"""

import io
import re
import time

import arxiv
import requests
from pypdf import PdfReader
from smolagents import tool

# --- Limits (cost and safety guardrails) -------------------------------------
MAX_SEARCH_RESULTS = 5         # most papers one search may return
MAX_ABSTRACT_CHARS = 500       # abstracts are trimmed to keep prompts small
MAX_PDF_BYTES = 20_000_000     # refuse PDFs larger than 20 MB
MAX_PDF_PAGES = 12             # only the first pages of a paper are read
DEFAULT_PDF_CHARS = 6000       # default amount of paper text returned
HARD_PDF_CHARS = 12000         # the agent cannot ask for more than this

HEADERS = {"User-Agent": "arxiv-agent-lab/1.0 (CS480 course project)"}  # arXiv blocks anonymous requests

# One shared client, reused across searches, so the library enforces arXiv's polite
# 3-second spacing between requests even when the user clicks several times quickly.
# A fresh client per call would reset that spacing and invite HTTP 429 throttling.
_CLIENT = arxiv.Client(page_size=5, delay_seconds=3.0, num_retries=3)

# arXiv IDs come in two shapes: new style (2301.01234, 2301.01234v2) and
# old style (hep-th/9901001). Both are matched exactly and inside URLs.
_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ID_EXACT = re.compile(rf"^(?:{_NEW_ID}|{_OLD_ID})$")
_ID_IN_URL = re.compile(rf"arxiv\.org/(?:abs|pdf)/({_NEW_ID}|{_OLD_ID})")


def _clean(text) -> str:
    """Collapse every run of whitespace into a single space."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _guardrail(message: str) -> str:
    """Prefix a failure message so the agent's rules recognize it."""
    return f"GUARDRAIL: {message}"


def _clamp_int(value, default: int, low: int, high: int) -> int:
    """Coerce a model-supplied argument to an int inside [low, high]."""
    try:
        return max(low, min(int(value), high))
    except (TypeError, ValueError):
        return default


def _extract_arxiv_id(value) -> str:
    """Return the arXiv ID held in an ID or arxiv.org URL, or '' if there is none."""
    value = re.sub(r"\.pdf$", "", str(value).strip(), flags=re.I)
    match = _ID_IN_URL.search(value)
    return match.group(1) if match else (value if _ID_EXACT.match(value) else "")


# --- Run log: what the tools really returned during this run -----------------
# app.py checks the agent's final answer against this log and blocks any answer
# citing a paper no tool returned, which is how invented citations get caught.
RUN_LOG = {"ids": set(), "titles": set(), "papers": []}


def reset_run_log() -> None:
    """Forget the previous run before a new one starts."""
    for collection in RUN_LOG.values():
        collection.clear()


def normalize_id(arxiv_id) -> str:
    """Drop the version suffix, so 1706.03762v7 and 1706.03762 compare equal."""
    return re.sub(r"v\d+$", "", str(arxiv_id).strip())


def normalize_title(title) -> str:
    """Keep only lowercase letters and digits, for tolerant title matching."""
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def find_arxiv_ids(text) -> set:
    """Every arXiv ID mentioned anywhere in a block of text, versions removed."""
    ids = re.findall(rf"(?<![\d.])({_NEW_ID})(?![\d])", str(text))
    ids += re.findall(rf"arxiv\.org/(?:abs|pdf)/({_OLD_ID})", str(text))
    return {normalize_id(found) for found in ids}


def _remember(papers: list) -> None:
    """Record papers in the run log so their citations count as verified."""
    RUN_LOG["papers"].extend(papers)
    RUN_LOG["ids"].update(normalize_id(paper["arxiv_id"]) for paper in papers)
    RUN_LOG["titles"].update(normalize_title(paper["title"]) for paper in papers)


# Words dropped before building a search phrase: they are about the request
# ("recent papers on X") rather than the topic itself.
_STOP_WORDS = {"a", "an", "the", "on", "of", "for", "and", "in", "to", "with", "about",
               "towards", "papers", "paper", "recent", "latest", "new", "newest"}
_YEAR_HINT = re.compile(r"\b(after|since|from|before|in)\s+(19|20)\d{2}\b|\b(19|20)\d{2}\b", re.I)


def _query_plans(query: str) -> list:
    """Build arXiv queries from strictest to loosest.

    arXiv treats a plain multi-word query as loose keywords, which buries the
    relevant papers under unrelated ones. Searching for the exact phrase in
    titles and abstracts first fixes that; the looser plans are only reached
    when the strict one finds too few papers.
    """
    if re.search(r"\b(ti|abs|au|cat|all):|\b(AND|OR|ANDNOT)\b", query):
        return [query]  # the caller already wrote arXiv query syntax

    words = [word for word in re.findall(r"[A-Za-z0-9\-]+", _YEAR_HINT.sub(" ", query))
             if word.lower() not in _STOP_WORDS]
    if not words:
        return [query]

    phrase = " ".join(words)
    plans = [f'ti:"{phrase}" OR abs:"{phrase}"']          # 1. exact phrase
    if len(words) > 1:
        plans.append(" AND ".join(f"all:{w}" for w in words))  # 2. all words, anywhere
    plans.append(phrase)                                   # 3. plain keywords
    return plans


@tool
def search_arxiv(query: str, max_results: int = 3, recent: bool = False) -> list:
    """
    Searches arXiv for academic papers on a topic. Use this first for any
    request about finding papers. One call is normally enough: the tool already
    tries an exact-phrase search and then looser searches by itself.

    Args:
        query: Only the topic, as a short phrase (e.g., 'neural stochastic
            differential equations'). Do not add words like 'recent' or
            'after 2020'; use the `recent` argument for that.
        max_results: How many papers to return, from 1 to 5. Defaults to 3.
        recent: Set to True when the user asks for recent, latest or new
            papers. Results are then ordered newest first.

    Returns:
        A list of dictionaries, one per paper, with the keys 'title', 'authors'
        (comma-separated string), 'year', 'arxiv_id', 'url', and 'abstract'.
        An empty list means no papers matched. If the search itself failed, the
        list holds one dictionary with a single 'error' key. In both of those
        cases, tell the user what happened and never invent papers.
    """
    query = _clean(query)
    if not query:
        return [{"error": _guardrail("the search query was empty.")}]

    max_results = _clamp_int(max_results, 3, 1, MAX_SEARCH_RESULTS)
    newest_first = str(recent).strip().lower() in {"true", "1", "yes"}
    sort_by = arxiv.SortCriterion.SubmittedDate if newest_first else arxiv.SortCriterion.Relevance

    def fetch(plan: str) -> list:
        """One arXiv request for a query plan; may raise on a 429 or network error."""
        found, seen = [], set()
        for paper in _CLIENT.results(arxiv.Search(query=plan, max_results=max_results, sort_by=sort_by)):
            short = normalize_id(paper.get_short_id())
            if short in seen:
                continue
            seen.add(short)
            abstract = _clean(paper.summary)
            found.append({
                "title": _clean(paper.title),
                "authors": ", ".join(author.name for author in paper.authors),
                "year": paper.published.year,
                "arxiv_id": paper.get_short_id(),
                "url": paper.entry_id,
                "abstract": (abstract[:MAX_ABSTRACT_CHARS].rstrip() + "..."
                             if len(abstract) > MAX_ABSTRACT_CHARS else abstract),
            })
            if len(found) >= max_results:
                break
        return found

    # Keep arXiv requests to a minimum: try the strict plan first and only fall back
    # to a looser one when a plan returns NOTHING (not merely fewer than max_results).
    # Firing fewer requests is what keeps us under arXiv's rate limit. On a 429 we wait
    # out the short penalty (up to ~30s total) and never pile on more requests.
    plans = _query_plans(query)
    for plan in plans:
        for attempt in range(2):
            try:
                papers = fetch(plan)
                if papers:
                    _remember(papers)
                    return papers
                break  # plan succeeded but found nothing -> try the next (looser) plan
            except Exception as exc:  # network errors, arXiv outages, bad responses
                if "429" in str(exc):
                    # One short retry only, so a throttled search fails fast instead of
                    # stalling the UI; a sustained throttle needs a wait or a new network.
                    if attempt == 0:
                        time.sleep(4)
                        continue
                    return [{"error": _guardrail(
                        "arXiv is rate-limiting requests (HTTP 429). Wait about a minute "
                        "without searching, then try again. Do not make up papers.")}]
                return [{"error": _guardrail(
                    f"the arXiv search failed ({type(exc).__name__}: {str(exc)[:200]}). "
                    "Report this failure to the user; do not make up papers.")}]
    return []  # every plan returned no matches


@tool
def read_paper_pdf(arxiv_id: str, max_chars: int = 6000) -> str:
    """
    Downloads a paper's PDF from arXiv and returns the text of its first pages.
    Use this when the user wants a summary of a paper's methodology or content
    that goes beyond the abstract. Read the returned text, then write the
    summary yourself.

    Args:
        arxiv_id: The arXiv ID (e.g., '1706.03762' or '1706.03762v7') or an
            arxiv.org URL. Take it from the 'arxiv_id' field of search_arxiv
            results; never guess an ID.
        max_chars: The most characters of text to return. Defaults to 6000,
            capped at 12000.

    Returns:
        The extracted paper text as a string. If the ID is invalid or the paper
        could not be downloaded or read, the string starts with 'GUARDRAIL:' and
        explains why. In that case, tell the user and do not summarize from
        memory.
    """
    paper_id = _extract_arxiv_id(arxiv_id)
    if not paper_id:
        return _guardrail(f"'{arxiv_id}' is not a valid arXiv ID. Expected a form like "
                          "'1706.03762' or 'hep-th/9901001'. Do not guess; ask the user or search first.")
    max_chars = _clamp_int(max_chars, DEFAULT_PDF_CHARS, 500, HARD_PDF_CHARS)

    try:
        response = requests.get(f"https://arxiv.org/pdf/{paper_id}", headers=HEADERS,
                                timeout=30, stream=True)
        if response.status_code == 404:
            return _guardrail(f"no paper exists on arXiv with ID '{paper_id}'. Tell the user it was not found.")
        if response.status_code != 200:
            return _guardrail(f"arXiv returned HTTP {response.status_code} for '{paper_id}'. "
                              "The paper could not be downloaded; report this instead of guessing.")

        # Stream the download so an oversized PDF is abandoned part-way instead
        # of being held in memory in full.
        data = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            data.extend(chunk)
            if len(data) > MAX_PDF_BYTES:
                return _guardrail(f"the PDF for '{paper_id}' is larger than "
                                  f"{MAX_PDF_BYTES // 1_000_000} MB and was not read.")
        if not bytes(data[:1024]).lstrip().startswith(b"%PDF"):
            return _guardrail(f"arXiv did not return a PDF for '{paper_id}' "
                              "(the paper may not exist or may have no PDF).")

        # Read pages only until max_chars is reached: a paper's method section
        # is near the front, and the rest would just cost tokens.
        reader = PdfReader(io.BytesIO(bytes(data)))
        pages, length = [], 0
        for page in reader.pages[:MAX_PDF_PAGES]:
            pages.append(_clean(page.extract_text() or ""))
            length += len(pages[-1])
            if length >= max_chars:
                break

        text = "\n".join(page for page in pages if page)
        if not text:
            return _guardrail(f"the PDF for '{paper_id}' contains no extractable text "
                              "(it may be a scanned document).")

        truncated = len(text) > max_chars or len(reader.pages) > MAX_PDF_PAGES
        text = text[:max_chars]
        RUN_LOG["ids"].add(normalize_id(paper_id))   # this paper was really read, and so
        RUN_LOG["ids"].update(find_arxiv_ids(text))  # were the papers it cites in that text
        note = f" Text is truncated to the first {max_chars} characters." if truncated else ""
        return f"[arXiv:{paper_id}, {len(reader.pages)} pages.{note}]\n{text}"
    except Exception as exc:  # timeouts, connection errors, corrupt PDFs
        return _guardrail(f"could not read the PDF for '{paper_id}' ({type(exc).__name__}: "
                          f"{str(exc)[:200]}). Report this failure; do not summarize from memory.")


@tool
def format_bibtex(title: str, authors: str, year: int, url: str) -> str:
    """
    Formats one paper as a BibTeX entry. Always use this for BibTeX instead of
    writing the entry yourself, and pass values exactly as search_arxiv
    returned them.

    Args:
        title: The paper title.
        authors: The authors as a comma-separated string (e.g., 'Ada Lovelace, Alan Turing').
        year: The publication year (e.g., 2024).
        url: The paper's arXiv URL (e.g., 'http://arxiv.org/abs/1706.03762v7').

    Returns:
        A BibTeX @article entry as a string, or a message starting with
        'GUARDRAIL:' if a required field is missing.
    """
    title = _clean(title)
    names = [name for name in (_clean(part) for part in str(authors).split(",")) if name]
    if not title or not names:
        return _guardrail("cannot build BibTeX without both a title and at least one author.")

    year_text = _clean(year)
    if not re.fullmatch(r"\d{4}", year_text):
        return _guardrail(f"'{year}' is not a valid four-digit year.")

    # Citation key in the usual style: first author's surname + year + first
    # meaningful title word, e.g. vaswani2017attention.
    surname = re.sub(r"[^A-Za-z]", "", names[0].split()[-1]).lower() or "paper"
    keywords = [word for word in re.findall(r"[A-Za-z]+", title) if word.lower() not in _STOP_WORDS]
    key = f"{surname}{year_text}{(keywords[0] if keywords else 'untitled').lower()}"

    fields = [f"  title = {{{title}}},",
              f"  author = {{{' and '.join(names)}}},",
              f"  year = {{{year_text}}},"]
    eprint = normalize_id(_extract_arxiv_id(url))
    if eprint:
        fields += [f"  eprint = {{{eprint}}},", "  archivePrefix = {arXiv},"]
    fields.append(f"  url = {{{_clean(url)}}}")
    return "\n".join([f"@article{{{key},", *fields, "}"])
