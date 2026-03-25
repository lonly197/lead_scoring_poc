from scripts.train_ohab import _candidate_prefixes_for_family


def test_candidate_prefixes_accepts_gbdt_alias():
    assert _candidate_prefixes_for_family("gbdt") == (
        "LightGBM",
        "LightGBMXT",
        "LightGBMLarge",
    )


def test_candidate_prefixes_accepts_lightgbm_alias():
    assert _candidate_prefixes_for_family("lightgbm") == (
        "LightGBM",
        "LightGBMXT",
        "LightGBMLarge",
    )
