"""One-off helper: force every \\begin{figure}[X] whose body references
llm/ figures to use [h!] placement so the figure stays near its
discussion in the report."""
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"(\\begin\{figure\})\[[^\]]*\]")
END = r"\end{figure}"

for path in [r'C:\Users\emir2\Desktop\CS349K_proj\monopoly\report\report.tex',
             r'C:\Users\emir2\Desktop\CS349K_proj\report\report.tex']:
    src = Path(path).read_text(encoding='utf-8')
    out_parts = []
    pos = 0
    n = 0
    for m in PATTERN.finditer(src):
        out_parts.append(src[pos:m.start()])
        block_end = src.find(END, m.end())
        block = src[m.end():block_end] if block_end > 0 else ''
        if 'llm/' in block:
            out_parts.append(m.group(1) + '[h!]')
            n += 1
        else:
            out_parts.append(m.group(0))
        pos = m.end()
    out_parts.append(src[pos:])
    Path(path).write_text(''.join(out_parts), encoding='utf-8')
    print(f'{path}: {n} LLM figures -> [h!]')
