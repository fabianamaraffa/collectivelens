"""The web app (Gradio 6). Layout and wiring only: the logic lives in the other modules."""

from __future__ import annotations

import logging

import gradio as gr

from . import __version__
from .ask import ask
from .budget import DailyBudget
from .charts import library_map, org_bar, topic_bar, year_topic_heatmap
from .config import Settings
from .evaluation import evaluate_topics
from .llm import make_client
from .render import (answer_html, groups_html, lang_label, results_html, status_html, table_frame)
from .runtime import Runtime, load_runtime
from .search import search

log = logging.getLogger(__name__)

RESULT_LIMIT = 10
ANY = "Any"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
    font=[gr.themes.Font("system-ui"), gr.themes.Font("-apple-system"), "sans-serif"],
)

# Colours come from Gradio's own variables, so the layout follows the light or dark theme.
CSS = """
.gradio-container { max-width: 1180px !important; --cl-muted: color-mix(in srgb, var(--body-text-color) 70%, var(--background-fill-primary)); }
.cl-hero { padding: 18px 4px 6px; }
.cl-hero h1 { font-family: "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif; font-weight: 700;
  font-size: clamp(1.7rem, 4vw, 2.3rem); margin: 0 0 4px; letter-spacing: -0.01em; color: var(--body-text-color); }
.cl-hero p { margin: 0; color: var(--cl-muted); font-size: 1.02rem; max-width: 60ch; }
.cl-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0 8px; }
.cl-badge { font-size: .8rem; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--border-color-primary);
  color: var(--cl-muted); background: var(--background-fill-secondary); }
.cl-badge.cl-ok { border-color: color-mix(in srgb, #008300 55%, transparent); color: var(--body-text-color); }
.cl-badge.cl-low { border-color: color-mix(in srgb, #eda100 70%, transparent); color: var(--body-text-color); }
.cl-info, .cl-warn { padding: 10px 14px; border-radius: 10px; margin: 8px 0; font-size: .92rem; line-height: 1.45;
  border: 1px solid var(--border-color-primary); color: var(--body-text-color); }
.cl-info { background: color-mix(in srgb, #2a78d6 8%, transparent); border-color: color-mix(in srgb, #2a78d6 35%, transparent); }
.cl-warn { background: color-mix(in srgb, #eda100 12%, transparent); border-color: color-mix(in srgb, #eda100 55%, transparent); }
.cl-count { color: var(--cl-muted); font-size: .9rem; margin: 4px 2px 10px; }
.cl-empty { padding: 28px 16px; text-align: center; color: var(--cl-muted);
  border: 1px dashed var(--border-color-primary); border-radius: 12px; }
.cl-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px 18px; margin: 0 0 12px;
  background: var(--block-background-fill); }
.cl-card-head { display: flex; gap: 12px; align-items: flex-start; }
.cl-rank { flex: none; width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; font-size: .78rem;
  font-weight: 600; background: var(--background-fill-secondary); color: var(--cl-muted); margin-top: 2px; }
.cl-card h3 { margin: 0; font-size: 1.08rem; line-height: 1.3; color: var(--body-text-color); }
.cl-card h3 a { color: inherit; text-decoration: underline; text-decoration-color: var(--border-color-primary); text-underline-offset: 3px; }
.cl-meta { color: var(--cl-muted); font-size: .86rem; margin-top: 2px; }
.cl-match { display: flex; align-items: center; gap: 10px; margin: 12px 0 2px 38px; font-size: .8rem; color: var(--cl-muted); }
.cl-match span { min-width: 118px; }
.cl-bar { flex: 1; max-width: 220px; height: 5px; border-radius: 3px; background: var(--background-fill-secondary); overflow: hidden; }
.cl-bar div { height: 100%; background: #2a78d6; border-radius: 3px; }
.cl-tags { margin: 10px 0 0 38px; }
.cl-chip { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; padding: 2px 10px; border-radius: 999px;
  background: var(--background-fill-secondary); color: var(--body-text-color); }
.cl-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.cl-muted { color: var(--cl-muted); }
.cl-summary { margin: 10px 0 0 38px; color: var(--body-text-color); line-height: 1.5; font-size: .94rem; }
.cl-card details { margin: 10px 0 0 38px; }
.cl-card summary { cursor: pointer; font-size: .86rem; color: var(--cl-muted); }
.cl-card blockquote, .cl-src blockquote { margin: 8px 0 0; padding: 2px 0 2px 12px; font-size: .88rem; line-height: 1.5;
  border-left: 3px solid var(--border-color-primary); color: var(--body-text-color); }
mark { background: color-mix(in srgb, #eda100 38%, transparent); color: inherit; border-radius: 3px; padding: 0 2px; }
.cl-answer { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px 18px; margin: 8px 0 14px;
  background: var(--block-background-fill); font-size: 1rem; line-height: 1.6; color: var(--body-text-color); }
.cl-answer p { margin: 0 0 .7em; } .cl-answer p:last-child { margin-bottom: 0; }
.cl-cite { display: inline-grid; place-items: center; min-width: 18px; height: 18px; padding: 0 4px; margin: 0 1px; border-radius: 9px;
  font-size: .7rem; font-weight: 600; vertical-align: 2px; background: color-mix(in srgb, #2a78d6 18%, transparent); color: var(--body-text-color); }
.cl-h4 { margin: 18px 2px 8px; font-size: .95rem; color: var(--cl-muted); font-weight: 600; }
.cl-src { display: flex; gap: 12px; padding: 12px 14px; border: 1px solid var(--border-color-primary); border-radius: 12px; margin-bottom: 8px;
  background: var(--block-background-fill); }
.cl-src.cl-dim { opacity: .72; }
.cl-num { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; font-size: .75rem; font-weight: 600;
  background: color-mix(in srgb, #2a78d6 18%, transparent); color: var(--body-text-color); }
.cl-src-title { font-size: .92rem; font-weight: 600; color: var(--body-text-color); }
.cl-used { font-size: .72rem; font-weight: 600; padding: 1px 7px; border-radius: 999px; margin-left: 4px;
  background: color-mix(in srgb, #008300 20%, transparent); color: var(--body-text-color); }
.cl-group { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; background: var(--block-background-fill); }
.cl-group ul { margin: 6px 0 0; padding-left: 18px; font-size: .9rem; } .cl-group li { margin: 2px 0; }
.cl-note { font-size: .86rem; color: var(--cl-muted); line-height: 1.5; }
footer { opacity: .7; }
[data-testid="dataframe"] * { font-family: var(--font) !important; }
@media (max-width: 640px) {
  .cl-match, .cl-tags, .cl-summary, .cl-card details { margin-left: 0; }
  .cl-card { padding: 14px; }
}
"""

ABOUT = """
### What this is
CollectiveLens helps people in organisations find their way in a collection of policy documents: search by
meaning, ask a question and get an answer with sources, and see how the collection is organised.

### How it works
1. **Passages.** Each document is cut into passages of about 180 words, and every passage is turned into a vector
   (a list of numbers that places texts about similar things close together). Version 1 embedded each whole
   document, but its model only reads the first part, so most of every document was ignored.
2. **Search.** Your question is turned into a vector and compared with all passages. Results show the best
   matching passages, and the match bar is relative to the best result (it is not a probability).
3. **Ask the library.** The best passages go to Claude with one instruction: answer only from them and cite
   them as [1], [2]. Citations that do not point to a real passage are removed. If the passages do not answer the
   question, Claude is told to say so.
4. **Topics.** Each document gets the closest of eight policy areas (the list is in `categories.yaml`). The
   number shown is a **similarity**, not a confidence, and "close to" flags documents that sit between two topics.
5. **Groups.** Documents are grouped automatically (k-means, number of groups chosen by silhouette score) and each
   group is described with the words that set it apart.

### Good to know
* **Privacy.** When you ask a question or build summaries with Claude, the question and the retrieved passages (or
  document text) are sent to Anthropic's API. Search, topics and charts run on this machine. A key you paste here is
  used for your session only and is not stored.
* **Languages.** With the semantic model a question in one language can find documents in another. With keyword
  search (shown in the status line) only shared words match.
* **Errors happen.** Topics and answers can be wrong. Check the cited passages before relying on an answer.
"""


def _filters(topics, orgs, langs, y_from, y_to) -> dict:
    def year(v):
        return int(v) if v and v != ANY else None

    return dict(topics=topics or None, orgs=orgs or None, languages=langs or None,
                year_from=year(y_from), year_to=year(y_to))


def build_app(runtime: Runtime | None = None) -> gr.Blocks:
    rt = runtime or load_runtime()
    lib = rt.library
    settings = rt.settings
    budget = DailyBudget()
    year_choices = [ANY] + [str(y) for y in lib.years()]
    lang_choices = [(lang_label(c), c) for c in lib.languages()]
    server_claude = rt.has_claude and budget.limit > 0

    def safe(fn):
        def wrapper(*args):
            try:
                return fn(*args)
            except Exception as exc:  # keep the app alive and tell the person what happened
                log.exception("Request failed")
                return f'<div class="cl-warn"><b>Something went wrong:</b> {type(exc).__name__}: {str(exc)[:200]}</div>'
        return wrapper

    @safe
    def do_search(query, sort, topics, orgs, langs, y_from, y_to):
        hits = search(lib, query, sort="newest" if sort == "Newest first" else "relevance", limit=RESULT_LIMIT,
                      **_filters(topics, orgs, langs, y_from, y_to))
        return results_html(lib, hits, query or "", limit=RESULT_LIMIT, total=len(lib.docs))

    @safe
    def do_ask(question, topics, orgs, langs, y_from, y_to, user_key):
        client, extra = None, ""
        if (user_key or "").strip():
            client = make_client(user_key)
        elif rt.has_claude:
            if budget.take():
                client = rt.client
            else:
                extra = " The shared key has reached its daily limit; paste your own key in the Written answers section above for a written answer."
        answer = ask(lib, question, client, model=settings.claude_model, k=settings.passages_for_answer,
                     max_tokens=settings.answer_max_tokens, **_filters(topics, orgs, langs, y_from, y_to))
        if extra and answer.mode == "passages":
            answer.note += extra
        return answer_html(answer, question)

    def do_browse(topics, orgs, langs, y_from, y_to):
        hits = search(lib, "", limit=len(lib.docs), **_filters(topics, orgs, langs, y_from, y_to))
        return table_frame(lib, [h.doc_index for h in hits])

    with gr.Blocks(title="CollectiveLens") as demo:
        gr.HTML('<div class="cl-hero"><h1>CollectiveLens</h1>'
                "<p>Search, ask and explore a library of policy documents, in several languages.</p></div>")
        gr.HTML(status_html(lib, claude_on=server_claude, claude_model=settings.claude_model, warning=rt.warning,
                            server_key=rt.has_claude))

        with gr.Accordion("Filters (topic, organisation, language, year)", open=False):
            with gr.Row():
                f_topics = gr.Dropdown(lib.topic_names, multiselect=True, label="Topic")
                f_orgs = gr.Dropdown(lib.orgs(), multiselect=True, label="Organisation")
                f_langs = gr.Dropdown(lang_choices, multiselect=True, label="Language")
            with gr.Row():
                f_from = gr.Dropdown(year_choices, value=ANY, label="From year")
                f_to = gr.Dropdown(year_choices, value=ANY, label="To year")
            gr.Markdown("Filters apply to Search, Ask the library and Browse.", elem_classes="cl-note")
        filters = [f_topics, f_orgs, f_langs, f_from, f_to]
        with gr.Accordion("Written answers: use your own Anthropic API key (optional)", open=False):
            key_box = gr.Textbox(type="password", label="Anthropic API key", placeholder="sk-ant-...",
                                 info="Kept for this session only and never saved. Your question and the matching "
                                      "passages are sent to Anthropic to write the answer.")

        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row():
                    q = gr.Textbox(label="What are you looking for?", scale=5, lines=1,
                                   placeholder="e.g. how can small organisations access EU funding?")
                    sort = gr.Radio(["Best match", "Newest first"], value="Best match", label="Order", scale=2)
                go = gr.Button("Search", variant="primary")
                if lib.is_sample:
                    gr.Examples(["how can small organisations access EU funding?", "support for students' mental health",
                                 "salute mentale studenti", "droits des jeunes et vote à seize ans"], inputs=q, label="Try")
                out = gr.HTML(results_html(lib, search(lib, "", limit=RESULT_LIMIT), "", limit=RESULT_LIMIT,
                                           total=len(lib.docs)))
                gr.on([go.click, q.submit], do_search, [q, sort] + filters, out)

            with gr.Tab("Ask the library"):
                aq = gr.Textbox(label="Your question", lines=2,
                                placeholder="e.g. What do the documents recommend to make exchange programmes fairer?")
                ago = gr.Button("Ask", variant="primary")
                if lib.is_sample:
                    gr.Examples(["What do the documents recommend to make exchange programmes fairer?",
                                 "Which changes to EU funding rules would help small organisations?",
                                 "Quali misure propongono per il benessere psicologico degli studenti?"],
                                inputs=aq, label="Try")
                aout = gr.HTML('<div class="cl-empty">Ask a question. The answer uses only passages from the library and '
                               "cites them.</div>")
                ago.click(do_ask, [aq] + filters + [key_box], aout)
                aq.submit(do_ask, [aq] + filters + [key_box], aout)

            with gr.Tab("Browse"):
                gr.Markdown("All documents that match the filters. **Similarity** is how close a document is to its topic "
                            "description. \"close to\" names a second topic when the choice was nearly a tie.")
                table = gr.Dataframe(do_browse([], [], [], ANY, ANY), interactive=False, wrap=True,
                                     column_widths=["30%", "20%", "7%", "10%", "25%", "8%"])
                gr.on([f.change for f in filters], do_browse, filters, table, show_progress="hidden")
                gr.Markdown("### Groups found automatically")
                gr.HTML(groups_html(lib))

            with gr.Tab("Insights"):
                gr.Markdown("Charts for the whole library (filters do not apply here).")
                gr.Plot(topic_bar(lib), show_label=False)
                gr.Plot(year_topic_heatmap(lib), show_label=False)
                gr.Plot(org_bar(lib), show_label=False)
                gr.Plot(library_map(lib), show_label=False)
                gr.Markdown("*Documents about similar things sit close together. Squeezing many dimensions into two loses "
                            "information, so treat distances as a rough guide."
                            + ("" if lib.semantic else " In keyword mode the map mostly separates languages, because "
                               "documents in the same language share more word parts.") + "*")
                report = evaluate_topics(lib)
                if report is not None:
                    with gr.Accordion("How well do topics match a person's choice?", open=False):
                        caveat = ("\n\n*On this demo library the documents were written to fit the topics, so this is only a "
                                  "smoke test, not evidence of accuracy.*" if lib.is_sample else "")
                        gr.Markdown(f"Reference topics were given for **{report.n}** documents. Top-1 agreement: "
                                    f"**{report.accuracy:.0%}**. Top-2 agreement: **{report.top2_accuracy:.0%}**." + caveat)
                        gr.Dataframe(report.per_topic.round(2).reset_index(), interactive=False)

            with gr.Tab("About"):
                gr.Markdown(ABOUT)
                gr.Markdown(f"CollectiveLens {__version__}. The first version (a notebook from the AllWomen bootcamp) is kept "
                            "in the `legacy/` folder of the repository.")
    return demo


def launch_kwargs(*, share: bool | None = None, port: int | None = None) -> dict:
    settings = Settings.from_env()
    kwargs = dict(theme=THEME, css=CSS, share=settings.share if share is None else share, show_error=True,
                  footer_links=["gradio"])
    if port:
        kwargs["server_port"] = port
    return kwargs


def launch(*, share: bool | None = None, port: int | None = None) -> None:
    build_app().launch(**launch_kwargs(share=share, port=port))
