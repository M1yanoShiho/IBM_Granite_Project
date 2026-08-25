from evidence_rag.materializer.answer_bank import AnswerBank, build_answer_bank, string_class


def test_string_class_partitions_by_mechanical_type() -> None:
    assert string_class("18") == "integer"
    assert string_class("2025") == "year"
    assert string_class("45.3") == "decimal"
    assert string_class("2019-01-05") == "date"
    assert string_class("Alice") == "name-1"
    assert string_class("New York") == "name-2"
    assert string_class("chocolate") == "noun-1"


def test_select_is_same_class_excludes_gold_and_deterministic() -> None:
    bank = build_answer_bank(["18", "23", "44", "2025", "Alice"], seed=42)
    chosen = bank.select(
        gold_value="18", gold_aliases=("18%", "18 percent"), string_class="integer"
    )
    assert chosen in {"23", "44"}
    assert bank.select("18", ("18%",), "integer") == chosen


def test_select_blocks_canonical_equivalents() -> None:
    bank = build_answer_bank(["1200000000", "5"], seed=42)
    assert bank.select("$1.2B", (), "integer") == "5"


def test_select_returns_none_when_no_alternative() -> None:
    bank = build_answer_bank(["18"], seed=42)
    assert bank.select("18", (), "integer") is None


def test_content_hash_is_stable() -> None:
    a = build_answer_bank(["18", "23"], seed=42)
    b = build_answer_bank(["23", "18"], seed=42)
    assert a.content_hash == b.content_hash
    assert build_answer_bank(["18", "23"], seed=7).content_hash != a.content_hash


def test_answer_bank_type_is_exposed() -> None:
    assert isinstance(build_answer_bank(["18"], seed=42), AnswerBank)
