from pathlib import Path

import pytest

from field_digitization.config import (
    ConfigError,
    FormLayout,
    load_layout,
    load_semantics,
    load_thresholds,
)


def test_default_thresholds_load():
    thresholds = load_thresholds()
    assert thresholds.ingest.render_dpi == 200
    assert thresholds.quality.min_blank_paper_level >= thresholds.quality.min_mean_brightness
    assert thresholds.quality.ink_relative_threshold < thresholds.quality.mark_relative_threshold
    assert thresholds.quality.max_noise_sigma > 0 and thresholds.quality.max_speckle_ratio > 0


def test_removed_blank_ratio_setting_is_rejected(tmp_path):
    # The area-based blank rule was replaced by content-component detection.
    path = tmp_path / "t.yaml"
    path.write_text("quality:\n  blank_max_mark_ratio: 0.002\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_default_semantics_load_and_have_both_text_and_choice_fields():
    # Replaces the test of the placeholder fixed-fraction layout file (approved): the form definition
    # now lives in coordinate-free form_semantics.yaml.
    sem = load_semantics()
    text_kinds = {f.kind for f in sem.header_text_fields} | {sem.section_title.kind, sem.comments.kind}
    assert text_kinds == {"printed", "handwritten"}
    assert sem.checklist.options == ["S", "R", "I", "N"]
    assert sem.expectation.options == ["Yes", "No", "N/A"] and len(sem.expectation.question_keys) == 4
    assert len(sem.location.options) == 10 and sem.shift.options == ["Day", "Night"]


def test_default_semantics_contain_no_coordinates_and_no_row_count():
    text = (Path(__file__).resolve().parents[1] / "config" / "form_semantics.yaml").read_text(encoding="utf-8")
    for forbidden in ("bbox", "x0", "y0", "row_count", "rows:", "slots"):
        assert forbidden not in text


def test_policy_decisions_required_and_optional():
    sem = load_semantics()
    assert sem.checklist.required and sem.expectation.required and sem.location.required and sem.shift.required
    assert all(f.required for f in sem.header_text_fields)
    assert sem.comments.required is False


def test_semantics_reject_duplicate_options_keys_and_unknown_fields(tmp_path):
    base = (Path(__file__).resolve().parents[1] / "config" / "form_semantics.yaml").read_text(encoding="utf-8")
    cases = {
        "dup_option": base.replace('options: ["S", "R", "I", "N"]', 'options: ["S", "S", "I", "N"]'),
        "dup_key": base.replace("key: shift", "key: location"),
        "unknown": base + "\nsurprise: 1\n",
        "one_option": base.replace('options: ["Day", "Night"]', 'options: ["Day"]'),
    }
    for name, text in cases.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_semantics(path)


def test_missing_semantics_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_semantics(tmp_path / "nope.yaml")


def test_the_placeholder_layout_file_is_gone_and_load_layout_needs_a_path():
    assert not (Path(__file__).resolve().parents[1] / "config" / "form_layout.yaml").exists()
    with pytest.raises(TypeError):
        load_layout()  # there is no default layout file any more


def test_alignment_and_ocr_defaults_load_and_orb_is_not_enabled_by_default():
    th = load_thresholds()
    assert th.alignment.detectors == ["sift"]
    assert th.alignment.allow_homography is True
    assert th.ocr.printed_min_confidence == pytest.approx(0.80)
    assert th.ocr.tesseract_cmd is None


@pytest.mark.parametrize("bad", ["detectors: []", "detectors: [sift, sift]", "detectors: [akaze]", "min_scale: 6.0",
                                 "min_inliers: 2", "ratio_test: 1.2", "max_shear_degrees: 120"])
def test_invalid_alignment_settings_raise(tmp_path, bad):
    path = tmp_path / "t.yaml"
    path.write_text(f"alignment:\n  {bad}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_orb_must_be_explicitly_listed(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("alignment:\n  detectors: [sift, orb]\n", encoding="utf-8")
    assert load_thresholds(path).alignment.detectors == ["sift", "orb"]


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_thresholds(tmp_path / "nope.yaml")


def test_unknown_threshold_key_raises(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("quality:\n  not_a_setting: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_blank_paper_level_below_dark_limit_raises(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "quality:\n  min_blank_paper_level: 50\n  min_mean_brightness: 90\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_thresholds(path)


@pytest.mark.parametrize(
    "setting", ["max_noise_sigma: 0", "max_speckle_ratio: 0", "min_content_extent_ratio: 0"]
)
def test_non_positive_noise_and_content_settings_raise(tmp_path, setting):
    path = tmp_path / "t.yaml"
    path.write_text(f"quality:\n  {setting}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_mark_threshold_must_be_above_ink_threshold(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "quality:\n  ink_relative_threshold: 0.9\n  mark_relative_threshold: 0.8\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_thresholds(path)


def test_invalid_bbox_raises(tmp_path):
    path = tmp_path / "l.yaml"
    path.write_text(
        "form_name: x\nfields:\n  - {name: a, kind: text, bbox: [0.5, 0.1, 0.4, 0.2]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_layout(path)


def test_duplicate_field_names_raise():
    with pytest.raises(ValueError):
        FormLayout(
            form_name="x",
            fields=[
                {"name": "a", "kind": "text", "bbox": (0.1, 0.1, 0.2, 0.2)},
                {"name": "a", "kind": "checkbox", "bbox": (0.3, 0.3, 0.4, 0.4)},
            ],
        )


def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_layout(path)
