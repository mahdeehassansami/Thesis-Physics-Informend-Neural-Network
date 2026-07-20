from pathlib import Path


path = Path(__file__).resolve().parents[1] / "configs" / "colab_experiments.json"
text = path.read_text(encoding="utf-8")
old = '"folder": "3rd_test"'
new = '"folder": "3rd_test/txt"'
if old not in text:
    raise RuntimeError("Expected IMS third-test path was not found.")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Updated IMS third-test path.")
