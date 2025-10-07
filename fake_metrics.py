from src.save_metrics import save_metrics

# Dummy object to simulate model.fit() output
class DummyHistory:
    history = {
        "accuracy": [0.92],
        "val_accuracy": [0.90],
        "loss": [0.15],
        "val_loss": [0.18]
    }

save_metrics(DummyHistory())
