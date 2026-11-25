# VoiceAgent launcher
# Pipeline: Silero VAD -> Parakeet TDT (STT) -> OpenCode Go (LLM brain) -> Kokoro-82M (TTS)
#
# Usage from the repository root:
#   powershell -ExecutionPolicy Bypass -File .\run-voice-agent.ps1          # mic + speakers (local)
#   powershell -ExecutionPolicy Bypass -File .\run-voice-agent.ps1 serve    # server only (ws://127.0.0.1:8765/v1/realtime)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$null = Set-Location $RepoRoot
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$S2SRoot = Join-Path $RepoRoot "speech-to-speech"
$TtsBenchRoot = Join-Path $RepoRoot "tts-bench"
$env:HF_HOME    = Join-Path $RuntimeRoot "cache\hf"
$env:TORCH_HOME = Join-Path $RuntimeRoot "cache\torch"
$env:NLTK_DATA  = Join-Path $RuntimeRoot "cache\nltk"
$env:PYTHONUTF8 = "1"

# OpenCode Go brain (OpenAI-compatible endpoint; session header required for routing)
$env:S2S_LLM_EXTRA_HEADERS = '{"x-opencode-session":"voice-agent"}'
$authPath = Join-Path $env:USERPROFILE ".local\share\opencode\auth.json"
if (-not (Test-Path $authPath)) {
    throw "OpenCode credentials not found at $authPath"
}
$opencodeKey = (Get-Content $authPath -Raw | ConvertFrom-Json).'opencode-go'.key

$exe  = Join-Path $S2SRoot ".venv\Scripts\speech-to-speech.exe"
if (-not (Test-Path $exe)) {
    throw "speech-to-speech is not installed at $exe"
}

# Voice selection, in order:
#   1. $env:VOICEAGENT_KOKORO_VOICE (a stock Kokoro voice name such as bm_george, or a .pt path)
#   2. tts-bench\models\kokoro\voices\custom.pt, if you created one (see README, "Voices")
#   3. the stock Kokoro voice bm_fable, downloaded from hexgrad/Kokoro-82M on first use
$customVoice = Join-Path $TtsBenchRoot "models\kokoro\voices\custom.pt"
if ($env:VOICEAGENT_KOKORO_VOICE) {
    $voice = $env:VOICEAGENT_KOKORO_VOICE
    if ($voice.EndsWith(".pt") -and -not (Test-Path $voice)) {
        throw "VOICEAGENT_KOKORO_VOICE points to a missing voicepack: $voice"
    }
} elseif (Test-Path $customVoice) {
    $voice = $customVoice
} else {
    $voice = "bm_fable"
    Write-Host "No custom voicepack at $customVoice; using the stock Kokoro voice bm_fable."
}
Write-Host "TTS voice: $voice"
$mode = if ($args.Count -gt 0) { $args[0] } else { "local" }

$common = @(
    "--stt", "parakeet-tdt",
    "--parakeet_tdt_device", "cuda",
    "--parakeet_tdt_language", "en",
    "--llm_backend", "chat-completions",
    "--responses_api_base_url", "https://opencode.ai/zen/go/v1",
    "--responses_api_api_key", $opencodeKey,
    "--model_name", "deepseek-v4.1-flash",
    "--responses_api_reasoning_effort", "none",
    "--responses_api_stream",
    "--tts", "kokoro",
    "--kokoro_device", "cuda",
    "--kokoro_voice", $voice,
    "--kokoro_lang_code", "b",
    "--kokoro_speed", "0.95",
    "--num_pipelines", "2"
)

# Add --local_audio_block_mic_during_playback to the local command if speaker
# feedback interrupts replies (also disables barge-in while the agent speaks).
& $exe $mode @common
