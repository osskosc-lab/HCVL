from pathlib import Path
import hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
fr=json.loads((ROOT/"preregistration/phase1.freeze.json").read_text())
actual=hashlib.sha256((ROOT/fr["file"]).read_bytes()).hexdigest()
if actual != fr["sha256"]:
    print("FAIL preregistration hash mismatch", file=sys.stderr)
    raise SystemExit(1)
print("PASS preregistration freeze", actual)
