import sys
import types


sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

from config.config import get_excluded_columns


def test_get_excluded_columns_includes_canonical_followup_text_fields():
    excluded = get_excluded_columns("线索评级结果")

    assert "首触跟进记录" in excluded
    assert "非首触跟进记录" in excluded
    assert "跟进记录_JSON" in excluded
    assert "跟进详情_JSON" in excluded
