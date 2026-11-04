"""HTML for cards, answers and the status line. Pure functions, so they can be tested without Gradio.

Everything that comes from a document or from the user is escaped before it is placed in HTML.
"""

from __future__ import annotations

import re
from html import escape as _esc

import pandas as pd

from .ask import Answer
from .charts import topic_color
from .library import Library
from .search import Hit, highlight

LANG_NAMES = {"en": "English", "it": "Italiano", "fr": "Français", "de": "Deutsch", "es": "Español",
              "nl": "Nederlands", "pt": "Português", "pl": "Polski", "ro": "Română", "unknown": "?"}

CLOSE_CALL_RATIO = 0.15  # a topic is a "close call" when its lead over the second topic is under 15% of its score


def esc(value: object) -> str:
    return _esc(str(value if value is not None else ""), quote=True)


def lang_label(code: str) -> str:
    return LANG_NAMES.get(code, code.upper())


def is_close_call(topic) -> bool:
    return topic.score > 0 and topic.margin < CLOSE_CALL_RATIO * topic.score


def topic_chip(lib: Library, name: str, *, close_to: str | None = None) -> str:
    extra = f' <span class="cl-muted">/ also {esc(close_to)}</span>' if close_to else ""
    return (f'<span class="cl-chip"><span class="cl-dot" style="background:{topic_color(lib, name)}"></span>'
            f"{esc(name)}{extra}</span>")


def _meta_line(hit_doc) -> str:
    parts = [esc(hit_doc.org)]
    if hit_doc.year:
        parts.append(str(hit_doc.year))
    parts.append(esc(lang_label(hit_doc.language)))
    return " &middot; ".join(parts)


def result_card(lib: Library, hit: Hit, query: str, position: int) -> str:
    doc = hit.doc
    topic = lib.topics[hit.doc_index]
    title = esc(doc.title)
    if doc.source_url.startswith(("http://", "https://")):
        title = f'<a href="{esc(doc.source_url)}" target="_blank" rel="noopener noreferrer">{title}</a>'
    bar = ""
    if query.strip():
        pct = round(hit.relative * 100)
        label = "Relative match" if lib.semantic else "Keyword match"
        bar = (f'<div class="cl-match" title="Score relative to the best result. 100% means the best match, '
               f'not a probability of being right."><span>{label} {pct}%</span>'
               f'<div class="cl-bar"><div style="width:{pct}%"></div></div></div>')
    summary = f'<p class="cl-summary">{esc(doc.summary)}</p>' if doc.summary else ""
    passages = ""
    if hit.passages and query.strip():
        items = "".join(f'<blockquote>{highlight(p.text, query)}</blockquote>' for p in hit.passages)
        passages = f"<details><summary>Best matching passages ({len(hit.passages)})</summary>{items}</details>"
    close = topic.second if is_close_call(topic) else None
    return (f'<article class="cl-card"><div class="cl-card-head"><span class="cl-rank">{position}</span>'
            f'<div><h3>{title}</h3><div class="cl-meta">{_meta_line(doc)}</div></div></div>'
            f'{bar}<div class="cl-tags">{topic_chip(lib, topic.name, close_to=close)}</div>'
            f'{summary}{passages}</article>')


def results_html(lib: Library, hits: list[Hit], query: str, *, limit: int, total: int | None = None) -> str:
    if not hits:
        return ('<div class="cl-empty"><b>No documents found.</b> Try other words, a different language, '
                "or loosen the filters above.</div>")
    if query.strip():
        shown = f" (showing the top {limit})" if len(hits) >= limit else ""
        label = f"{len(hits)} document{'s' if len(hits) != 1 else ''} for &ldquo;{esc(query.strip())}&rdquo;{shown}"
    else:
        of = f" of {total}" if total and total > len(hits) else ""
        label = f"{len(hits)}{of} document{'s' if (total or len(hits)) != 1 else ''}, newest first. Type a question above to search."
    head = f'<div class="cl-count">{label}</div>'
    return head + "".join(result_card(lib, h, query, i) for i, h in enumerate(hits, start=1))


def _paragraphs(text: str) -> str:
    safe = esc(text)
    safe = re.sub(r"\[(\d{1,2})\]", r'<span class="cl-cite">\1</span>', safe)
    return "".join(f"<p>{p.strip()}</p>" for p in re.split(r"\n\s*\n", safe) if p.strip()) or f"<p>{safe}</p>"


def answer_html(answer: Answer, question: str) -> str:
    parts = []
    if answer.note:
        kind = "cl-warn" if answer.mode == "claude" else "cl-info"
        parts.append(f'<div class="{kind}">{esc(answer.note)}</div>')
    if answer.text:
        parts.append(f'<div class="cl-answer">{_paragraphs(answer.text)}</div>')
    if answer.sources:
        heading = "Sources" if answer.mode == "claude" else "Best matching passages"
        cards = []
        for s in answer.sources:
            used = s.n in answer.cited
            state = ""
            if answer.mode == "claude":
                state = ' <span class="cl-used">cited</span>' if used else ' <span class="cl-muted">retrieved, not cited</span>'
            year = f", {s.doc.year}" if s.doc.year else ""
            cards.append(
                f'<div class="cl-src{"" if used or answer.mode != "claude" else " cl-dim"}">'
                f'<span class="cl-num">{s.n}</span><div><div class="cl-src-title">{esc(s.doc.title)}'
                f' <span class="cl-muted">({esc(s.doc.org)}{year})</span>{state}</div>'
                f'<blockquote>{highlight(s.passage.text, "", limit=520)}</blockquote></div></div>')
        parts.append(f'<h4 class="cl-h4">{heading}</h4>' + "".join(cards))
    return "".join(parts)


def status_html(lib: Library, *, claude_on: bool, claude_model: str, warning: str | None, server_key: bool) -> str:
    chips = [
        f'<span class="cl-badge">{len(lib.docs)} documents</span>',
        f'<span class="cl-badge">{len(lib.languages())} language{"s" if len(lib.languages()) != 1 else ""}: '
        f'{esc(", ".join(lang_label(c) for c in lib.languages()))}</span>',
        f'<span class="cl-badge {"cl-ok" if lib.semantic else "cl-low"}">'
        f'{"Search by meaning" if lib.semantic else "Keyword search"}</span>',
        f'<span class="cl-badge {"cl-ok" if claude_on else "cl-low"}">'
        f'{"Claude on: " + esc(claude_model) if claude_on else "Claude off (add a key for written answers)"}</span>',
    ]
    out = ['<div class="cl-badges">' + "".join(chips) + "</div>"]
    if lib.is_sample:
        out.append(f'<div class="cl-info"><b>Demo library.</b> These {len(lib.docs)} documents are fictional, written only to show '
                   "how CollectiveLens works, in English, Italian and French. Build your own library with "
                   "<code>python -m collectivelens build</code> (see the README).</div>")
    if warning:
        out.append(f'<div class="cl-warn">{esc(warning)}</div>')
    return "".join(out)


def groups_html(lib: Library) -> str:
    """The groups found automatically, with the words that set each apart."""
    members: dict[int, list[int]] = {}
    for i, c in enumerate(lib.cluster_ids):
        members.setdefault(int(c), []).append(i)
    blocks = []
    for cid in sorted(members, key=lambda c: -len(members[c])):
        terms = ", ".join(lib.cluster_terms.get(cid, [])[:6]) or "(no distinguishing words)"
        titles = "".join(f"<li>{esc(lib.docs[i].title)} <span class=\"cl-muted\">({esc(lib.topics[i].name)})</span></li>"
                         for i in members[cid])
        blocks.append(f'<div class="cl-group"><div class="cl-group-head"><b>Group of {len(members[cid])}</b>'
                      f'<span class="cl-muted"> &middot; {esc(terms)}</span></div><ul>{titles}</ul></div>')
    return "".join(blocks)


def table_frame(lib: Library, indices: list[int]) -> pd.DataFrame:
    rows = []
    for i in indices:
        d, t = lib.docs[i], lib.topics[i]
        rows.append({
            "Title": d.title, "Organisation": d.org, "Year": d.year or "", "Language": lang_label(d.language),
            "Topic": f"{t.name} (close to {t.second})" if is_close_call(t) else t.name,
            "Similarity": round(t.score, 3),
        })
    return pd.DataFrame(rows, columns=["Title", "Organisation", "Year", "Language", "Topic", "Similarity"])
