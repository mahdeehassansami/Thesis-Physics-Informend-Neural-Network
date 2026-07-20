from pathlib import Path


path = Path(__file__).resolve().parent / "validate_colab_workflow.py"
text = path.read_text(encoding="utf-8")
old = '"total_cycles": 400000.0'
new = '"cycles_per_time_unit": 400000.0'
if old not in text:
    raise RuntimeError("Expected validation metadata field was not found.")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Updated Run 2 validation metadata.")
