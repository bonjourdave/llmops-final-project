from pathlib import Path

from ingestion.loaders import load_netflix_csv


_HEADER = (
    "show_id,type,title,director,cast,country,"
    "date_added,release_year,rating,duration,listed_in,description\n"
)


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows), encoding="utf-8")
    return path


def test_returns_one_record_per_row(tmp_path):
    csv = _write_csv(
        tmp_path / "netflix.csv",
        [
            's1,Movie,Alpha,,,,,2020,PG,90 min,Dramas,First description.',
            's2,TV Show,Beta,,,,,2021,TV-MA,1 Season,Comedies,Second description.',
        ],
    )
    records = load_netflix_csv(csv)
    assert len(records) == 2


def test_composite_text_contains_all_fields(tmp_path):
    csv = _write_csv(
        tmp_path / "netflix.csv",
        ['s1,Movie,Test Title,,,,,2020,PG,90 min,Dramas,A great drama.'],
    )
    records = load_netflix_csv(csv)
    text = records[0]["text"]
    assert "Test Title" in text
    assert "Movie" in text
    assert "Dramas" in text
    assert "A great drama." in text


def test_composite_text_uses_pipe_separator(tmp_path):
    csv = _write_csv(
        tmp_path / "netflix.csv",
        ['s1,Movie,T,,,,,2020,PG,90 min,L,D'],
    )
    records = load_netflix_csv(csv)
    assert records[0]["text"] == "T | Movie | L | D"


def test_raw_fields_preserved(tmp_path):
    csv = _write_csv(
        tmp_path / "netflix.csv",
        ['s99,TV Show,My Show,,,,,2022,TV-14,2 Seasons,Sci-Fi,Space stuff.'],
    )
    rec = load_netflix_csv(csv)[0]
    assert rec["show_id"] == "s99"
    assert rec["title"] == "My Show"
    assert rec["type"] == "TV Show"
    assert rec["listed_in"] == "Sci-Fi"
    assert rec["description"] == "Space stuff."


def test_missing_optional_fields_dont_raise(tmp_path):
    # Minimal row — only show_id present, everything else empty
    path = tmp_path / "netflix.csv"
    path.write_text(
        "show_id,type,title,listed_in,description\ns1,,,,\n",
        encoding="utf-8",
    )
    records = load_netflix_csv(path)
    assert len(records) == 1
    assert records[0]["text"] == " |  |  | "


def test_empty_file_returns_empty_list(tmp_path):
    path = tmp_path / "netflix.csv"
    path.write_text(_HEADER, encoding="utf-8")
    assert load_netflix_csv(path) == []
