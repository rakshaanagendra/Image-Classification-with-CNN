import json
import os

def save_metrics(history, output_path=None):
    """
    Save training metrics (accuracy, loss) as JSON to the project root.
    """
    if history is None or not hasattr(history, "history"):
        print("⚠️ No valid history object provided. Metrics not saved.")
        return

    # Build path to project root (two levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = output_path or os.path.join(project_root, "metrics.json")

    try:
        metrics = {
            "train_accuracy": float(history.history.get("accuracy", [0])[-1]),
            "val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
            "train_loss": float(history.history.get("loss", [0])[-1]),
            "val_loss": float(history.history.get("val_loss", [0])[-1]),
        }

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"✅ Metrics saved successfully to {output_path}")

    except Exception as e:
        print(f"❌ Error saving metrics: {e}")
