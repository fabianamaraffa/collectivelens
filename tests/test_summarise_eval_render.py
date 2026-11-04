import json

from collectivelens.ask import Answer, Source
from collectivelens.evaluation import evaluate_topics
from collectivelens.library import Document, Library
from collectivelens.render import answer_html, groups_html, result_card, results_html, status_html, table_frame
from collectivelens.search import Passage, search
from collectivelens.summarise import Summariser, load_provided_summaries
from conftest import FakeClient


def test_provided_summary_wins_and_no_call_is_made():
    client = FakeClient()
    s = Summariser(client, provided={"a.txt": "Ready-made."})
    r = s.summarise("a.txt", "A", "text")
    assert r.text == "Ready-made." and r.source == "provided" and not client.calls


def test_summary_is_cached(tmp_path):
    client = FakeClient("A short summary.")
    cache = tmp_path / "cache.json"
    first = Summariser(client, cache_path=cache).summarise("a.txt", "A", "some text")
    again = Summariser(client, cache_path=cache).summarise("a.txt", "A", "some text")
    assert first.source.startswith("claude:") and again.source == "cache" and len(client.calls) == 1
    Summariser(client, cache_path=cache).summarise("a.txt", "A", "changed text")
    assert len(client.calls) == 2


def test_no_client_gives_no_summary():
    assert Summariser(None).summarise("a.txt", "A", "x").source == "none"


def test_failed_summary_is_not_cached(tmp_path):
    s = Summariser(FakeClient(error=RuntimeError("x")), cache_path=tmp_path / "c.json")
    assert s.summarise("a.txt", "A", "x").source == "none" and not (tmp_path / "c.json").exists()


def test_both_summary_file_formats(tmp_path):
    d = tmp_path / "d.json"
    d.write_text(json.dumps({"a.pdf": "One"}))
    l = tmp_path / "l.json"
    l.write_text(json.dumps([{"filename": "b.pdf", "summary": "Two"}, {"filename": "c.pdf", "summary": "Summary unavailable"}]))
    assert load_provided_summaries(d) == {"a.pdf": "One"} and load_provided_summaries(l) == {"b.pdf": "Two"}
    assert load_provided_summaries(None) == {}


def test_topic_evaluation_on_sample(sample_lib):
    report = evaluate_topics(sample_lib)
    assert report.n == 13 and 0 <= report.accuracy <= 1 and report.confusion.shape == (8, 8)
    assert "Top-1 agreement" in report.to_markdown()


def test_evaluation_reports_disagreements(sample_lib, categories):
    docs = [Document(**{**d.__dict__}) for d in sample_lib.docs]
    docs[0].gold_topic = "Peace & Security" if sample_lib.topics[0].name != "Peace & Security" else "Human Rights"
    lib = Library(docs, type(sample_lib.embedder)(), categories)
    report = evaluate_topics(lib)
    assert len(report.wrong) >= 1 and report.accuracy < 1


def test_evaluation_needs_gold_topics(sample_lib, categories):
    docs = [Document(**{**d.__dict__, "gold_topic": ""}) for d in sample_lib.docs]
    assert evaluate_topics(Library(docs, type(sample_lib.embedder)(), categories)) is None


def test_cards_escape_untrusted_text(sample_lib):
    evil = '<img src=x onerror=alert(1)>'
    hit = search(sample_lib, "youth participation")[0]
    hit.doc = Document(**{**hit.doc.__dict__, "title": evil, "summary": evil, "org": evil, "source_url": "javascript:alert(1)"})
    html = result_card(sample_lib, hit, "youth", 1)
    assert "<img" not in html and "javascript:" not in html.split("href=")[-1] if "href=" in html else True
    assert "&lt;img" in html


def test_answer_html_escapes_and_marks_citations(sample_lib):
    doc = sample_lib.docs[0]
    src = Source(1, doc, Passage(0, 0, "Passage <b>text</b>", 0.5))
    html = answer_html(Answer("Claim <script>x</script> [1] and [7].", [src], [1], "claude"), "claim")
    assert "<script>" not in html and '<span class="cl-cite">1</span>' in html
    assert "&lt;b&gt;" in html


def test_status_and_groups_and_table(sample_lib):
    html = status_html(sample_lib, claude_on=False, claude_model="m", warning="note <b>", server_key=False)
    assert "13 documents" in html and "Demo library" in html and "note &lt;b&gt;" in html
    assert "Group of" in groups_html(sample_lib)
    frame = table_frame(sample_lib, list(range(13)))
    assert len(frame) == 13 and list(frame.columns)[-1] == "Similarity"


def test_no_results_message(sample_lib):
    assert "No documents found" in results_html(sample_lib, [], "x", limit=10)
