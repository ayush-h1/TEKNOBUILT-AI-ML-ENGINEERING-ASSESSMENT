import pytest

from field_digitization.config import (
    ConfigError,
    FieldKind,
    FormLayout,
    load_layout,
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


def test_default_layout_loads_and_has_both_kinds():
    layout = load_layout()
    kinds = {f.kind for f in layout.fields}
    assert kinds == {FieldKind.TEXT, FieldKind.CHECKBOX}


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
