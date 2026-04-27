"""Sanity check IEEE main.tex: citations, figure paths, env balance, word count."""
import re
from collections import Counter
from pathlib import Path

ROOT = Path("reports/paper/ieee_en")
tex = (ROOT / "main.tex").read_text(encoding="utf-8")
bib = (ROOT / "bib.bib").read_text(encoding="utf-8")

cited = re.findall(r"\\cite\{([^}]+)\}", tex)
cite_keys = set()
for grp in cited:
    for k in grp.split(","):
        cite_keys.add(k.strip())

defined = set(re.findall(r"@\w+\{([^,]+),", bib))
print(f"Citations in tex: {len(cite_keys)}")
print(f"Entries in bib:   {len(defined)}")
print(f"MISSING in bib:   {sorted(cite_keys - defined) or 'NONE'}")
print(f"Unused in tex:    {sorted(defined - cite_keys) or 'NONE'}")

figs = re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
print(f"\nFigure refs ({len(figs)}):")
for f in figs:
    p = ROOT / f
    print(f"  {f}  ->  {'OK' if p.exists() else 'MISSING'}")

beg = Counter(re.findall(r"\\begin\{(\w+)\}", tex))
end = Counter(re.findall(r"\\end\{(\w+)\}", tex))
extra_b = beg - end
extra_e = end - beg
print(f"\nbegin/end balanced: {'YES' if not extra_b and not extra_e else f'NO (extra begin={dict(extra_b)}, extra end={dict(extra_e)})'}")

words = len(re.findall(r"[A-Za-z]+", tex))
print(f"\nWord-like tokens: {words}  (rough est ~{words // 700} pages of body in 2-col conf format)")
