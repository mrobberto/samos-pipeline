# tools/audit_pipeline_config_usage.py

from pathlib import Path
import ast
import re

ROOT = Path.cwd().resolve().parent
CFG = ROOT / "config" / "pipeline_config.py"

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".ipynb_checkpoints",
    "htmlcov", ".pytest_cache",
}

EXTS = {".py", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".json"}

tree = ast.parse(CFG.read_text())

names = set()
for node in tree.body:
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id.isupper():
                names.add(t.id)

names = sorted(names)

files = []
for p in ROOT.rglob("*"):
    if any(part in EXCLUDE_DIRS for part in p.parts):
        continue
    if p.is_file() and p.suffix.lower() in EXTS:
        files.append(p)

for name in names:
    pat = re.compile(rf"\b{re.escape(name)}\b")
    hits = []

    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue

        if pat.search(txt):
            hits.append(f.relative_to(ROOT))

    print("\n" + "=" * 80)
    print(name)

    if hits:
        for h in sorted(set(hits)):
            print("   ", h)
    else:
        print("    NONE")