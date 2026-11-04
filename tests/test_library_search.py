from collectivelens.library import Document, Library, load_documents, save_documents
from collectivelens.search import highlight, search


def test_sample_library_has_topics_and_groups(sample_lib):
    assert len(sample_lib.docs) == 13
    assert len(sample_lib.topics) == 13 and len(sample_lib.cluster_ids) == 13
    assert all(t.name in sample_lib.topic_names for t in sample_lib.topics)
    assert all(len(sample_lib.cluster_terms[c]) >= 0 for c in set(sample_lib.cluster_ids))
    assert sample_lib.is_sample and not sample_lib.semantic


def test_every_document_is_split_into_passages(sample_lib):
    assert all(len(idx) >= 2 for idx in sample_lib.doc_chunks)


def test_save_and_load_roundtrip(tmp_path, sample_lib):
    save_documents(sample_lib.docs, tmp_path, is_sample=True, name="x")
    docs, meta = load_documents(tmp_path)
    assert [d.id for d in docs] == [d.id for d in sample_lib.docs]
    assert meta["is_sample"] is True and meta["documents"] == 13


def test_missing_library_gives_a_helpful_error(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="build"):
        load_documents(tmp_path)


def test_search_finds_the_right_document(sample_lib):
    hits = search(sample_lib, "small organisations EU funding lump sums")
    assert hits[0].doc.title == "Simpler funding for small organisations"
    assert hits[0].relative == 1.0 and all(0 <= h.relative <= 1 for h in hits)
    assert hits[0].passages and hits[0].passages[0].score >= hits[0].passages[-1].score


def test_search_in_italian(sample_lib):
    assert search(sample_lib, "salute mentale studenti")[0].doc.language == "it"


def test_filters(sample_lib):
    hits = search(sample_lib, "", orgs=["Demo Youth Network"])
    assert {h.doc.org for h in hits} == {"Demo Youth Network"} and len(hits) == 2
    hits = search(sample_lib, "", year_from=2024, year_to=2024)
    assert hits and all(h.doc.year == 2024 for h in hits)
    assert search(sample_lib, "", languages=["fr"]) and all(h.doc.language == "fr" for h in search(sample_lib, "", languages=["fr"]))
    assert search(sample_lib, "youth", topics=["Climate Justice"], limit=10)[0].topic == "Climate Justice"


def test_empty_query_lists_newest_first(sample_lib):
    years = [h.doc.year for h in search(sample_lib, "", limit=13)]
    assert years == sorted(years, reverse=True)


def test_nonsense_query_returns_nothing(sample_lib):
    assert search(sample_lib, "zzzzqqqq xxxxwwww") == []


def test_newest_sort(sample_lib):
    hits = search(sample_lib, "youth", sort="newest", limit=10)
    years = [h.doc.year for h in hits]
    assert years == sorted(years, reverse=True)


def test_highlight_escapes_html_and_marks_words():
    out = highlight("Use <script>alert(1)</script> for youth participation & more", "youth participation")
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<mark>youth</mark>" in out and "<mark>participation</mark>" in out
    assert "&amp;" in out  # the entity is not broken by the highlighter
    assert "<mark>amp" not in highlight("a & b", "amp quot")


def test_tiny_library_still_builds(categories):
    from collectivelens.embedders import LiteEmbedder

    docs = [Document(id=f"d{i}", filename=f"{i}.txt", title=f"Doc {i}", org="O", year=2020 + i, language="en",
                     text="youth policy climate education " * 30 + f"unique{i}") for i in range(3)]
    lib = Library(docs, LiteEmbedder(), categories)
    assert len(lib.topics) == 3
