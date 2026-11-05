import csv


from collectivelens.ingest import ingest_folder, load_metadata, read_text

BODY = "Young people deserve a real say in decisions that affect them. " * 12


def test_ingest_uses_metadata_case_insensitively(tmp_path):
    (tmp_path / "a.txt").write_text(BODY, encoding="utf-8")
    (tmp_path / "b.md").write_text(BODY + " second", encoding="utf-8")
    (tmp_path / "tiny.txt").write_text("too short", encoding="utf-8")
    meta = tmp_path / "meta.csv"
    with open(meta, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["File_Name", "Title", "Member_Org", "Publication_Year", "Gold_Topic"])
        w.writerow(["a.pdf", "First", "Org A", "2023.0", "Human Rights"])
        w.writerow(["a.txt", "Duplicate row", "Org Z", "1999", "Other"])  # first row wins
    docs = {d.filename: d for d in ingest_folder(tmp_path, meta)}
    assert set(docs) == {"a.txt", "b.md"}  # the tiny file is skipped
    assert docs["a.txt"].title == "First" and docs["a.txt"].org == "Org A"
    assert docs["a.txt"].year == 2023 and docs["a.txt"].gold_topic == "Human Rights"
    assert docs["b.md"].org == "Unknown"


def test_ids_are_unique(tmp_path):
    for name in ("same.txt", "same.md"):
        (tmp_path / name).write_text(BODY, encoding="utf-8")
    ids = [d.id for d in ingest_folder(tmp_path)]
    assert len(ids) == len(set(ids)) == 2


def test_missing_metadata_columns_do_not_crash(tmp_path):
    meta = tmp_path / "m.csv"
    meta.write_text("something,else\n1,2\n", encoding="utf-8")
    assert load_metadata(meta) == {}


def test_reads_a_real_pdf(tmp_path):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 4))
    fig.text(0.1, 0.5, "Youth participation matters in every European policy area")
    pdf = tmp_path / "doc.pdf"
    fig.savefig(pdf, format="pdf")
    assert "participation" in read_text(pdf)


def test_a_broken_pdf_is_skipped_and_does_not_stop_the_build(tmp_path, caplog):
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    (tmp_path / "good.txt").write_text(BODY, encoding="utf-8")
    docs = ingest_folder(tmp_path)
    assert [d.filename for d in docs] == ["good.txt"]
    assert "broken.pdf" in caplog.text
