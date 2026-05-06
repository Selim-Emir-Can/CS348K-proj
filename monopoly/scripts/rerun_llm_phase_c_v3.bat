@echo off
setlocal

REM ----------------------------------------------------------------------
REM Re-run Task 1 (LLM-only Phase C eval) on the v3 GA-winner boards so
REM the LLM-side figures use the same physical boards as the rest of the
REM v3-consistent report.
REM
REM Differences from Apr 29 Phase C (logs/llm_eval/{2p,3p}_v2):
REM   - --ga-2p-jsonl logs/optimizer_v3/ga_2p_mask.jsonl  (was Apr 24)
REM   - --ga-3p-jsonl logs/optimizer_v3/ga_3p_mask.jsonl  (was Apr 24)
REM   - Output to logs/llm_eval/{2p,3p}_v3 (Apr 29 v2 dirs preserved as
REM     reference)
REM
REM Idempotent: each step is skipped if its output already exists.
REM Re-run after a partial failure to resume from where it stopped.
REM
REM Estimated wall time: ~6 hours on a 12 GB GPU
REM   (40 games per player count, ~6-10 s per LLM decision,
REM    ~50-100 decisions per game).
REM
REM Usage (from cmd.exe):
REM   monopoly\scripts\rerun_llm_phase_c_v3.bat
REM ----------------------------------------------------------------------

cd /d C:\Users\emir2\Desktop\CS349K_proj\monopoly
if errorlevel 1 (echo cd failed & exit /b 1)

call conda activate cs224r-proj
if errorlevel 1 (echo conda activate cs224r-proj failed & exit /b 1)

set "PYTHONPATH=."

mkdir logs\llm_eval\2p_v3 2>nul
mkdir logs\llm_eval\3p_v3 2>nul

set "SUMMARY=logs\llm_eval\summary_v3.log"
set "MODEL=models/qwen2.5-1.5B"
set "GA2P=logs/optimizer_v3/ga_2p_mask.jsonl"
set "GA3P=logs/optimizer_v3/ga_3p_mask.jsonl"

if not exist "%SUMMARY%" echo === Started %DATE% %TIME% (LLM Phase C on v3 boards) === > "%SUMMARY%"
echo === Resumed %DATE% %TIME% === >> "%SUMMARY%"

REM ===== 2-player Phase C eval =====
set "RUN=2p_v3_eval"
if exist "logs\llm_eval\2p_v3\summary.csv" (
    echo [%TIME%] skip    %RUN% ^(summary.csv exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/eval_llm_on_boards.py --boards default ga_2p_winner --n-players 2 --n-seeds 20 --out-dir logs/llm_eval/2p_v3 --model-name %MODEL% --ga-2p-jsonl %GA2P% --ga-3p-jsonl %GA3P% > "logs\llm_eval\2p_v3.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== 3-player Phase C eval =====
set "RUN=3p_v3_eval"
if exist "logs\llm_eval\3p_v3\summary.csv" (
    echo [%TIME%] skip    %RUN% ^(summary.csv exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/eval_llm_on_boards.py --boards default ga_3p_winner --n-players 3 --n-seeds 20 --out-dir logs/llm_eval/3p_v3 --model-name %MODEL% --ga-2p-jsonl %GA2P% --ga-3p-jsonl %GA3P% > "logs\llm_eval\3p_v3.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

REM ===== Decision-log analyses =====
set "RUN=analyze_2p_v3"
if exist "logs\llm_eval\2p_v3\analysis_default.md" (
    echo [%TIME%] skip    %RUN% ^(analysis_default.md exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/analyze_llm_decisions.py --in logs/llm_eval/2p_v3 > "logs\llm_eval\analyze_2p_v3.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

set "RUN=analyze_3p_v3"
if exist "logs\llm_eval\3p_v3\analysis_default.md" (
    echo [%TIME%] skip    %RUN% ^(analysis_default.md exists^) >> "%SUMMARY%"
) else (
    echo [%TIME%] starting %RUN% >> "%SUMMARY%"
    python scripts/analyze_llm_decisions.py --in logs/llm_eval/3p_v3 > "logs\llm_eval\analyze_3p_v3.log" 2>&1
    if errorlevel 1 goto :error
    echo [%TIME%] done    %RUN% >> "%SUMMARY%"
)

echo === Done %DATE% %TIME% === >> "%SUMMARY%"
echo.
echo Done. Summary: %SUMMARY%
echo.
echo Next step (after waking up): tell me "v3 LLM done" and I will
echo regenerate the LLM figures from logs/llm_eval/{2p,3p}_v3/ and
echo replace the PNGs in report/figures/llm/.
exit /b 0

:error
echo === FAILED at %RUN% ^(%DATE% %TIME%^) === >> "%SUMMARY%"
echo.
echo FAILED at %RUN%. Last lines of relevant log:
if "%RUN%"=="2p_v3_eval" powershell -NoProfile -Command "Get-Content 'logs\llm_eval\2p_v3.log' -Tail 30"
if "%RUN%"=="3p_v3_eval" powershell -NoProfile -Command "Get-Content 'logs\llm_eval\3p_v3.log' -Tail 30"
if "%RUN%"=="analyze_2p_v3" powershell -NoProfile -Command "Get-Content 'logs\llm_eval\analyze_2p_v3.log' -Tail 30"
if "%RUN%"=="analyze_3p_v3" powershell -NoProfile -Command "Get-Content 'logs\llm_eval\analyze_3p_v3.log' -Tail 30"
exit /b 1
