from src.utils import paths


def test_project_root_is_repo_root():
    assert (paths.PROJECT_ROOT / "requirements.txt").exists()


def test_inputs_default_under_data():
    assert paths.CORPUS_SOURCE_ROOT == paths.PROJECT_ROOT / "data" / "corpus_sources"
    assert paths.FM_ASSETS_ROOT == paths.PROJECT_ROOT / "data" / "foundation_models"


def test_cache_dir_defaults_under_project_root():
    assert paths.CACHE_DIR == paths.PROJECT_ROOT / "data" / "embeddings"
