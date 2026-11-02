# VoiceTutor: one-shot setup + verification on the G14.
# Run from the project folder in an Anaconda PowerShell prompt:
#   conda activate D:/conda_envs/fine_tune_env
#   powershell -ExecutionPolicy Bypass -File scripts/verify.ps1
# Writes everything to eval/results/verify_log.txt. Paste that file back to the assistant.

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force eval/results | Out-Null
$log = "eval/results/verify_log.txt"
Start-Transcript -Path $log -Force | Out-Null

Write-Host "== 1. Environment =="
python --version
python -c "import sys; print(sys.executable)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

Write-Host "== 2. Back up env, then install (no upgrades of existing packages such as torch) =="
python -m pip freeze > env_backup.txt
python -m pip install --upgrade-strategy only-if-needed -r requirements.txt
python -m pip check

Write-Host "== 3. CUDA for faster-whisper =="
# ctranslate2 needs cuBLAS + cuDNN 9 DLLs. If torch (CUDA build) is installed, reuse its DLL folder.
$torchLib = python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>$null
if ($torchLib -and (Test-Path $torchLib)) { $env:PATH = "$torchLib;$env:PATH"; Write-Host "Added torch DLLs: $torchLib" }
else { python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* }
python -c "import ctranslate2; print('CUDA devices for ctranslate2:', ctranslate2.get_cuda_device_count())"
$env:LOCAL_WHISPER_MODEL = "large-v3-turbo"
python -c "from voicetutor import asr; print('local whisper device:', asr.local_device())"

Write-Host "== 4. Unit tests =="
python -m pytest -q

Write-Host "== 5. Retrieval eval (real) =="
python eval/eval_retrieval.py

$haveRec = (Test-Path eval/audio/wer_25.wav) -and (Test-Path eval/audio/qa_25.wav)
$keySet = (Select-String -Path .env -Pattern '^GROQ_API_KEY=\S+' -Quiet -ErrorAction SilentlyContinue)
Write-Host "Recordings present: $haveRec   Groq key in .env: $keySet"

foreach ($mode in @("with-key", "no-key")) {
    if ($mode -eq "with-key" -and -not $keySet) { Write-Host "Skipping with-key run: no key in .env"; continue }
    Write-Host "== 6. QA + WER evals: $mode =="
    if ($mode -eq "no-key") { $env:GROQ_API_KEY = " " } else { Remove-Item Env:GROQ_API_KEY -ErrorAction SilentlyContinue }
    $env:METRICS_PATH = "logs/latency_$mode.jsonl"
    $env:RESULT_TAG = $mode            # result files get a _with-key / _no-key suffix
    python eval/eval_qa.py --text
    if ($haveRec) {
        python eval/eval_qa.py
        python eval/eval_wer.py
    } else {
        Write-Host "No recordings yet: running synthetic voice QA to check the pipeline (NOT a result)"
        python eval/eval_qa.py --synthetic
    }
}
Remove-Item Env:GROQ_API_KEY, Env:RESULT_TAG -ErrorAction SilentlyContinue

Write-Host "== 7. App starts (5 s smoke test) =="
$p = Start-Process python -ArgumentList "app.py" -PassThru -WindowStyle Hidden
Start-Sleep 25
try { (Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7860/ -TimeoutSec 10).StatusCode } catch { "App did not respond: $_" }
Stop-Process -Id $p.Id -Force

Write-Host "== Done. Now open the app yourself (python app.py) and test the microphone in the browser. =="
Stop-Transcript | Out-Null
