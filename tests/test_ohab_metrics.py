from src.evaluation.ohab_metrics import classification_report_dict, classification_report_text


def test_classification_report_helpers_use_zero_division_zero():
    y_true = ["H", "A", "B", "B"]
    y_pred = ["H", "A", "A", "A"]

    report_dict = classification_report_dict(y_true, y_pred)
    report_text = classification_report_text(y_true, y_pred)

    assert report_dict["B"]["precision"] == 0.0
    assert report_dict["B"]["recall"] == 0.0
    assert "B" in report_text
    assert "0.00" in report_text
