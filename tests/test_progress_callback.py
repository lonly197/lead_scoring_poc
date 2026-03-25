import types

from src.training.progress_callback import TrainingProgressCallback


class FakeTrainer:
    def __init__(self):
        self._callback_early_stop = False
        self.hyperparameters = {"GBM": [{}, {}], "CAT": {}}
        self.num_stack_levels = 1
        self.model_attributes = {
            "ModelA": {
                "val_score": 0.91,
                "fit_time": 12.5,
                "predict_time": 0.8,
            }
        }

    def get_model_attribute(self, model_name, attribute, default=None):
        return self.model_attributes.get(model_name, {}).get(attribute, default)

    def load_model(self, model_name):
        return types.SimpleNamespace(**self.model_attributes.get(model_name, {}))


def test_progress_callback_handles_autogluon_15_lifecycle():
    callback = TrainingProgressCallback(show_progress_bar=False, log_interval=1, time_limit=120)
    trainer = FakeTrainer()
    model = types.SimpleNamespace(name="ModelA")

    callback.before_trainer_fit(
        trainer,
        hyperparameters=trainer.hyperparameters,
        level_start=1,
        level_end=2,
    )
    early_stop, skip_model = callback.before_model_fit(
        trainer,
        model,
        time_limit=90,
        stack_name="core",
        level=1,
    )
    assert (early_stop, skip_model) == (False, False)

    trainer_stop = callback.after_model_fit(
        trainer,
        ["ModelA"],
        stack_name="core",
        level=1,
    )
    assert trainer_stop is False

    summary = callback.get_summary()
    assert summary["models"] == ["ModelA"]
    assert summary["best_model"] == "ModelA"
    assert summary["best_score"] == 0.91
    assert summary["estimated_total_models"] == 8


def test_progress_callback_handles_empty_model_names():
    callback = TrainingProgressCallback(show_progress_bar=False, log_interval=1)
    trainer = FakeTrainer()
    model = types.SimpleNamespace(name="ModelA")

    callback.before_trainer_fit(
        trainer,
        hyperparameters=trainer.hyperparameters,
        level_start=1,
        level_end=2,
    )
    callback.before_model_fit(
        trainer,
        model,
        time_limit=60,
        stack_name="core",
        level=1,
    )

    trainer_stop = callback.after_model_fit(
        trainer,
        [],
        stack_name="core",
        level=1,
    )

    assert trainer_stop is False
    assert callback.models_completed == []
