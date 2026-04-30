"""One-off helper: remove em-dashes from the report.tex copies.

We replace:
  - ' — '  -> '; '   (sentence-internal clause separator)
  - '—'    -> ', '   (rare no-space form)
  - ' --- '     -> '; '   (LaTeX em-dash markup)
  - '---'       -> ', '   (rare no-space LaTeX)
EN-dashes ('--') are preserved (used for numeric ranges like ``10--20'').
"""
from pathlib import Path

EM = "—"

PAIRS = [
    (' ' + EM + ' ', '; '),
    (EM,             ', '),
    (' --- ',        '; '),
    ('---',          ', '),
]

for path in [r'C:\Users\emir2\Desktop\CS349K_proj\monopoly\report\report.tex',
             r'C:\Users\emir2\Desktop\CS349K_proj\report\report.tex']:
    src = Path(path).read_text(encoding='utf-8')
    n_total = 0
    for old, new in PAIRS:
        cnt = src.count(old)
        n_total += cnt
        src = src.replace(old, new)
    Path(path).write_text(src, encoding='utf-8')
    print(f'{path}: replaced {n_total} em-dash occurrences')
