import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_model.py"
_spec = importlib.util.spec_from_file_location("fetch_model", SCRIPT)
fetch_model = importlib.util.module_from_spec(_spec)
sys.modules["fetch_model"] = fetch_model
_spec.loader.exec_module(fetch_model)


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def no_download(*args, **kwargs):
    raise AssertionError("nothing should be downloaded")


@pytest.fixture
def models_dir(tmp_path):
    return tmp_path / "models"


# ---------- choosing a source and a destination ----------


def test_plan_url_wins_over_hub_repo(models_dir):
    env = {"MODEL_URL": "https://example.com/m.zip", "MODEL_HF_REPO": "me/model"}
    plan = fetch_model.make_plan(env, models_dir)

    assert plan.source == "url"
    assert plan.dest == models_dir / fetch_model.FINETUNED_DIR_NAME


def test_plan_hub_repo_goes_to_the_finetuned_folder(models_dir):
    plan = fetch_model.make_plan({"MODEL_HF_REPO": "me/model"}, models_dir)

    assert plan.source == "hf"
    assert plan.dest == models_dir / fetch_model.FINETUNED_DIR_NAME


def test_plan_without_source_falls_back_to_pretrained(models_dir):
    plan = fetch_model.make_plan({}, models_dir)

    assert plan.source == "pretrained"
    assert plan.dest == models_dir / fetch_model.PRETRAINED_DIR_NAME


def test_plan_keeps_a_baked_in_finetuned_model(models_dir):
    baked = models_dir / fetch_model.FINETUNED_DIR_NAME
    baked.mkdir(parents=True)
    (baked / "config.json").write_text("{}")

    plan = fetch_model.make_plan({}, models_dir)

    assert plan.dest == baked


def test_plan_model_dir_override(models_dir, tmp_path):
    custom = tmp_path / "custom"

    without_source = fetch_model.make_plan({"REVIEWPULSE_MODEL_DIR": str(custom)}, models_dir)
    with_source = fetch_model.make_plan(
        {"REVIEWPULSE_MODEL_DIR": str(custom), "MODEL_HF_REPO": "me/model"}, models_dir
    )

    assert (without_source.dest, without_source.source) == (custom, "none")
    assert (with_source.dest, with_source.source) == (custom, "hf")


# ---------- nothing to do ----------


def test_run_does_nothing_when_a_model_is_in_place(models_dir, monkeypatch):
    present = models_dir / fetch_model.PRETRAINED_DIR_NAME
    present.mkdir(parents=True)
    (present / "config.json").write_text("{}")
    monkeypatch.setattr(fetch_model, "download_hf", no_download)
    monkeypatch.setattr(fetch_model, "download_file", no_download)

    assert fetch_model.run({}, models_dir) == 0


def test_run_fails_when_override_folder_is_empty_and_no_source(models_dir, tmp_path):
    env = {"REVIEWPULSE_MODEL_DIR": str(tmp_path / "empty")}

    assert fetch_model.run(env, models_dir) == 1


# ---------- zip download ----------


def test_zip_with_files_at_the_root_is_installed(models_dir, tmp_path):
    archive = make_zip(
        tmp_path / "m.zip",
        {"config.json": b"{}", "metadata.json": b'{"fine_tuned": true}', "weights.bin": b"w"},
    )

    code = fetch_model.run({"MODEL_URL": archive.as_uri()}, models_dir)

    dest = models_dir / fetch_model.FINETUNED_DIR_NAME
    assert code == 0
    assert (dest / "config.json").is_file()
    assert (dest / "weights.bin").read_bytes() == b"w"


def test_zip_with_a_wrapper_folder_is_flattened(models_dir, tmp_path):
    archive = make_zip(
        tmp_path / "m.zip",
        {"nlptown_finetuned_v2/config.json": b"{}", "nlptown_finetuned_v2/weights.bin": b"w"},
    )

    code = fetch_model.run({"MODEL_URL": archive.as_uri()}, models_dir)

    assert code == 0
    assert (models_dir / fetch_model.FINETUNED_DIR_NAME / "config.json").is_file()


def test_matching_sha256_is_accepted(models_dir, tmp_path):
    archive = make_zip(tmp_path / "m.zip", {"config.json": b"{}"})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    env = {"MODEL_URL": archive.as_uri(), "MODEL_SHA256": digest.upper()}

    assert fetch_model.run(env, models_dir) == 0


def test_wrong_sha256_fails_and_installs_nothing(models_dir, tmp_path):
    archive = make_zip(tmp_path / "m.zip", {"config.json": b"{}"})
    env = {"MODEL_URL": archive.as_uri(), "MODEL_SHA256": "0" * 64}

    code = fetch_model.run(env, models_dir)

    assert code == 1
    assert not (models_dir / fetch_model.FINETUNED_DIR_NAME).exists()
    assert [p.name for p in models_dir.iterdir()] == []  # staging folder cleaned up


def test_zip_with_path_traversal_is_rejected(models_dir, tmp_path):
    archive = make_zip(tmp_path / "m.zip", {"config.json": b"{}", "../evil.txt": b"x"})

    code = fetch_model.run({"MODEL_URL": archive.as_uri()}, models_dir)

    assert code == 1
    assert not (tmp_path / "evil.txt").exists()


def test_zip_without_config_json_is_rejected(models_dir, tmp_path):
    archive = make_zip(tmp_path / "m.zip", {"readme.txt": b"hello"})

    assert fetch_model.run({"MODEL_URL": archive.as_uri()}, models_dir) == 1


def test_url_scheme_is_restricted(tmp_path):
    with pytest.raises(ValueError):
        fetch_model.download_file("ftp://example.com/m.zip", tmp_path / "out.zip")


def test_redact_drops_the_query_string():
    url = "https://example.com/m.zip?token=secret#frag"

    assert fetch_model.redact(url) == "https://example.com/m.zip"


# ---------- Hugging Face Hub and pretrained fallback ----------


def test_hub_source_passes_repo_and_revision(models_dir, monkeypatch):
    calls = []

    def fake_download(repo_id, dest, revision):
        calls.append((repo_id, revision))
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(fetch_model, "download_hf", fake_download)
    env = {"MODEL_HF_REPO": "me/model", "MODEL_HF_REVISION": "abc123"}

    code = fetch_model.run(env, models_dir)

    assert code == 0
    assert calls == [("me/model", "abc123")]
    assert (models_dir / fetch_model.FINETUNED_DIR_NAME / "config.json").is_file()


def test_pretrained_fallback_writes_metadata(models_dir, monkeypatch):
    def fake_download(repo_id, dest, revision):
        assert repo_id == fetch_model.PRETRAINED_REPO
        (dest / "config.json").write_text("{}")

    monkeypatch.setattr(fetch_model, "download_hf", fake_download)

    code = fetch_model.run({}, models_dir)

    dest = models_dir / fetch_model.PRETRAINED_DIR_NAME
    assert code == 0
    assert '"fine_tuned": false' in (dest / "metadata.json").read_text()


def test_download_error_gives_exit_code_1_and_leaves_no_partial_model(models_dir, monkeypatch):
    def broken_download(repo_id, dest, revision):
        (dest / "half.bin").write_bytes(b"partial")
        raise OSError("connection reset")

    monkeypatch.setattr(fetch_model, "download_hf", broken_download)

    code = fetch_model.run({"MODEL_HF_REPO": "me/model"}, models_dir)

    assert code == 1
    assert not (models_dir / fetch_model.FINETUNED_DIR_NAME).exists()
    assert [p.name for p in models_dir.iterdir()] == []


# ---------- stays in step with the API ----------


def test_folder_names_match_the_api():
    bert_model = pytest.importorskip("src.bert_model")

    assert fetch_model.FINETUNED_DIR_NAME == bert_model.FINETUNED_DIR_NAME
    assert fetch_model.PRETRAINED_DIR_NAME == bert_model.PRETRAINED_DIR_NAME
    assert fetch_model.MODEL_DIR_ENV == bert_model.MODEL_DIR_ENV
