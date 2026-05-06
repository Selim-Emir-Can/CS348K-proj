@echo off
setlocal

REM ----------------------------------------------------------------------
REM Overnight re-run with a larger search budget AND a larger evaluation
REM sample size, to test whether the 2p GA-vs-random tie observed in the
REM May 6 run is a search-budget or sampling-noise artefact, or a real
REM property of the design space.
REM
REM Differences from the standard re-run:
REM   - GA: pop 30 x 30 generations (was pop 20 x 20 generations).
REM     Search budget = 30 + 29*28 = 842 evals (was 362).
REM   - Random: --iters 842 (matches GA budget; was 400).
REM   - --n-games 200 (was default 100 = 10 games per matchup).
REM     Now 20 games per matchup, halving per-candidate variance.
REM   - Output dir: logs\optimizer_v3 (logs\optimizer_v2 stays untouched
REM     so the May 6 short run remains as a comparison point).
REM
REM Idempotent: each step is skipped if its output file already exists.
REM Re-running this .bat after a partial failure resumes from where it
REM stopped. To force a step to re-run, delete its output file first.
REM
REM Estimated wall time: ~5 hours on a single CPU.
REM
REM Usage (from cmd.exe):
REM   monopoly\scripts\rerun_strategy_experiments_overnight.bat
REM ----------------------------------------------------------------------

cd /d C:\Users\emir2\Desktop\CS349K_proj\monopoly
if errorlevel 1 (echo cd failed & exit /b 1)

call conda activate cs224r-proj
if errorlevel 1 (echo conda activate cs224r-proj failed & exit /b 1)

set "PYTHONPATH=."

mkdir logs\optimizer_v3 2>nul

set "OUT=logs/optimizer_v3"
set "OUTBS=logs\optimizer_v3"
set "SUMMARY=%OUTBS%\summary.log"
set "BASE=--base-seed 42 --search-seed 0 --matchup-seed 1234 --max-turns 200 --removal-direction cheapest --target-rounds 60 --target-transfer 100 --n-games 200"
set "GAARGS=--search ga --pop 30 --generations 30 --elitism 2"

if not exist "%SUMMARY%" echo === Started %DATE% %TIME% (overnight: pop 30 x gens 30, n_games 200) === > "%SUMMARY%"
echo === Resumed %DATE% %TIME% === >> "%SUMMARY%"

REM ===== Random-search baselines =====
set "RUN=random_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 --search random --iters 842 %BASE% --w-fair 1.0 --w-fmax 0.5 --w-len 0.5 --w-draw 0.3 --w-money 0.3 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=random_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 --search random --iters 842 %BASE% --w-fair 1.0 --w-fmax 0.5 --w-len 0.5 --w-draw 0.3 --w-money 0.3 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== Combined-objective GA =====
set "RUN=ga_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 %GAARGS% %BASE% --w-fair 1.0 --w-fmax 0.5 --w-len 0.5 --w-draw 0.3 --w-money 0.3 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=ga_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 %GAARGS% %BASE% --w-fair 1.0 --w-fmax 0.5 --w-len 0.5 --w-draw 0.3 --w-money 0.3 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== Single-objective ablations =====
set "RUN=abl_fair_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 %GAARGS% %BASE% --w-fair 1.0 --w-fmax 0.0 --w-len 0.0 --w-draw 0.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_fair_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 %GAARGS% %BASE% --w-fair 1.0 --w-fmax 0.0 --w-len 0.0 --w-draw 0.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_len_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 1.0 --w-draw 0.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_len_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 1.0 --w-draw 0.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_draw_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 0.0 --w-draw 1.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_draw_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 0.0 --w-draw 1.0 --w-money 0.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_money_2p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 2 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 0.0 --w-draw 0.0 --w-money 1.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=abl_money_3p_mask"
if exist "%OUTBS%\%RUN%.jsonl" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/optimize_board.py --n-players 3 %GAARGS% %BASE% --w-fair 0.0 --w-fmax 0.0 --w-len 0.0 --w-draw 0.0 --w-money 1.0 --run-name %RUN% --out-dir %OUT% > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== High-confidence cross-evaluation =====
set "RUN=cross_eval_mask"
if exist "%OUTBS%\cross_eval_mask.json" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/cross_eval.py --runs %OUT%/ga_2p_mask.jsonl %OUT%/ga_3p_mask.jsonl --identity --n-games 1000 --out %OUT%/cross_eval_mask.json > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== 30x30 strategy heatmaps =====
set "RUN=heatmap_ga2p_mask"
if exist "%OUTBS%\heatmap_ga2p_mask.json" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/strategy_heatmap.py --runs %OUT%/ga_2p_mask.jsonl --identity-baseline --n-players 2 --n-games 20 --out %OUT%/heatmap_ga2p_mask > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=heatmap_ga3p_mask"
if exist "%OUTBS%\heatmap_ga3p_mask.json" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/strategy_heatmap.py --runs %OUT%/ga_3p_mask.jsonl --identity-baseline --n-players 3 --n-games 20 --out %OUT%/heatmap_ga3p_mask > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== Convergence + Pareto reports =====
set "RUN=reports_2p_mask"
if exist "%OUTBS%\reports_2p_mask\convergence.png" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/report_runs.py %OUT%/random_2p_mask.jsonl %OUT%/ga_2p_mask.jsonl %OUT%/abl_fair_2p_mask.jsonl %OUT%/abl_len_2p_mask.jsonl %OUT%/abl_draw_2p_mask.jsonl %OUT%/abl_money_2p_mask.jsonl --out-dir %OUT%/reports_2p_mask > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=reports_3p_mask"
if exist "%OUTBS%\reports_3p_mask\convergence.png" (
    echo [%TIME%] skip    %RUN% ^(exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/report_runs.py %OUT%/random_3p_mask.jsonl %OUT%/ga_3p_mask.jsonl %OUT%/abl_fair_3p_mask.jsonl %OUT%/abl_len_3p_mask.jsonl %OUT%/abl_draw_3p_mask.jsonl %OUT%/abl_money_3p_mask.jsonl --out-dir %OUT%/reports_3p_mask > "%OUTBS%\%RUN%.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

echo === Done %DATE% %TIME% === >> "%SUMMARY%"
echo.
echo Done. Summary: %SUMMARY%
echo Per-run logs in %OUTBS%\^<run_name^>.log
exit /b 0

:error
echo === FAILED at %RUN% ^(%DATE% %TIME%^) === >> "%SUMMARY%"
echo.
echo FAILED at %RUN%. Last lines of %OUTBS%\%RUN%.log:
powershell -NoProfile -Command "Get-Content '%OUTBS%\%RUN%.log' -Tail 25"
exit /b 1
