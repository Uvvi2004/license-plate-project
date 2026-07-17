"""Tests for plate-string validation, built from the real 4K-footage failures."""

from license_plate_pipeline.validation import canonical_plate, display_plate, is_valid_plate


def test_valid_us_and_euro_plates():
    for plate in ["6FVZ747", "C98191P", "7BW2396", "MF16JZ", "R-183-JF", "66-HH-07", "L-656-XH"]:
        assert is_valid_plate(plate), plate


def test_rejects_numeric_only_fragments():
    # "619879", "1073", "6755" etc - no letter.
    for junk in ["619879", "657648", "1073", "3789", "6755"]:
        assert not is_valid_plate(junk), junk


def test_rejects_alpha_only_state_name_misreads():
    # "IDAHO" state name misread as "HDAHO"/"YISNI" - no digit.
    for junk in ["HDAHO", "IDAU", "TDAIO", "YISNI", "SEP"]:
        assert not is_valid_plate(junk), junk


def test_rejects_too_short_or_too_long():
    assert not is_valid_plate("A1")       # too short
    assert not is_valid_plate("AB12CD34E")  # 9 chars, too long


def test_canonical_strips_separators_and_non_ascii():
    assert canonical_plate("R-183-JF") == "R183JF"
    assert canonical_plate("皖EKH9211") == "EKH9211"
    assert canonical_plate("lekh 92") == "LEKH92"


def test_display_strips_non_ascii_but_keeps_dashes():
    assert display_plate("R-183-JF") == "R-183-JF"
    assert display_plate("皖EKH9211") == "EKH9211"


def test_chinese_hallucination_becomes_valid_after_stripping():
    # "皖EKH9211" -> canonical "EKH9211" is a valid plate, so we recover it
    # rather than dropping the whole read.
    assert is_valid_plate("皖EKH9211")
