"""One-off CVPR-style restructure of the report.tex copies.

Three changes:
  1. Insert a brief "Related Work" section between Introduction and
     Problem Definition.
  2. Rename "Evaluation" -> "Experiments" (preserving \\label).
  3. Rename "Generalisation: lessons for multi-agent system design"
     -> "Discussion: lessons for multi-agent system design"
     (preserving \\label).
"""
from pathlib import Path

RELATED_WORK = r"""
\section{Related work}\label{sec:related}

\paragraph{Agents as design probes.} Using simulated agents to playtest
games is well established. Recent work has used reinforcement learning,
search agents, and hand-coded heuristics to surface balance issues in
both digital and tabletop games. Our contribution is not the use of
agents per se but the protocol for using \emph{multiple agent classes
with diverse strategy distributions} as cross-checks, with
disagreement-as-diagnostic and human playtesting reserved for the
boundary cases.

\paragraph{LLM-based agents.} Small instruction-tuned LLMs (Qwen2.5,
Llama 3) have been used as decision-making agents in board games,
simulations, and tool-use evaluations. Our setup uses a 1.5B model with
greedy decoding, a structured \texttt{STATE}/\texttt{ECHO}/\texttt{REASON}/\texttt{ANSWER}
prompt, and a deterministic per-field validator with a bounded retry
budget. The validator-and-retry stack is closer in spirit to
constrained decoding and self-consistency than to standard chain of
thought; we report failure modes specific to that protocol (Section
\ref{sec:phase_c}).

\paragraph{Genetic algorithms over environment design.} GAs and
evolutionary search are routine for environment design problems. Our
GA is unremarkable on its own (population 20, generations 10,
tournament selection, uniform crossover, Gaussian + bit-flip
mutation); the contribution lies in the evaluator: a 30-strategy
parametric pool, with a single-personality LLM evaluator as a
cross-class control.

\paragraph{Robust evaluation.} Variance reduction via shared random
numbers (CRN) is standard in stochastic simulation; per-objective
single-knob ablations are standard in optimisation. We adopt both
because they are cheap, reproducible, and orthogonal. The novelty in
this work is connecting them to the agent-as-design-probe framing.

"""


def insert_related_work(src: str) -> str:
    marker = r'\section{Problem definition}\label{sec:problem}'
    if marker not in src:
        return src
    if r'\section{Related work}' in src:
        return src   # idempotent
    return src.replace(marker, RELATED_WORK + marker, 1)


def rename(src: str, old: str, new: str) -> str:
    return src.replace(old, new, 1)


for path in [r'C:\Users\emir2\Desktop\CS349K_proj\monopoly\report\report.tex',
             r'C:\Users\emir2\Desktop\CS349K_proj\report\report.tex']:
    src = Path(path).read_text(encoding='utf-8')
    src = insert_related_work(src)
    src = rename(src,
                 r'\section{Evaluation}\label{sec:eval}',
                 r'\section{Experiments}\label{sec:eval}')
    src = rename(src,
                 r'\section{Generalisation: lessons for multi-agent system design}\label{sec:gen}',
                 r'\section{Discussion: lessons for multi-agent system design}\label{sec:gen}')
    Path(path).write_text(src, encoding='utf-8')
    print(f'{path}: restructure applied')
