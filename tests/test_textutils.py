from collectivelens.textutils import chunk_text, content_words, detect_language, normalize_key, slugify


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text(words(50)) == [words(50)]


def test_empty_text_has_no_chunks():
    assert chunk_text("   ") == []


def test_chunks_overlap_and_cover_everything():
    chunks = chunk_text(words(400), size=180, overlap=30)
    assert len(chunks) >= 3
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-30:] == second[:30]
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert seen == {f"w{i}" for i in range(400)}


def test_short_tail_is_folded_into_previous_chunk():
    chunks = chunk_text(words(190), size=180, overlap=30, min_tail=40)
    assert len(chunks) == 1 or all(len(c.split()) >= 40 for c in chunks)
    seen = set()
    for c in chunks:
        seen.update(c.split())
    assert len(seen) == 190


def test_normalize_key_drops_folders_and_extensions():
    assert normalize_key("pdfs/My Report.PDF") == "My Report"
    assert normalize_key("C:\\docs\\a.txt") == "a"
    assert normalize_key(None) == ""


def test_slugify_handles_accents():
    assert slugify("Éducation à la paix!") == "education-a-la-paix"


def test_language_detection_is_repeatable():
    text = "Le Réseau Démo Santé Mentale alerte sur le manque de soutien psychologique pour les jeunes."
    assert {detect_language(text) for _ in range(5)} == {"fr"}
    assert detect_language("hi") == "unknown"


def test_content_words_drop_stopwords():
    assert content_words("The youth and the participation") == ["youth", "participation"]
