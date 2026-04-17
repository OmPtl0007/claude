# start_claude_code.ps1
# Terminal 2: The CLI

$env:ANTHROPIC_BASE_URL = "http://localhost:5000"
$env:ANTHROPIC_API_KEY = "sk-ant-dummy"

# Map Claude internal roles to Gemma models
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = "gemma-4-26b-it"   # Planning & Architecture
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "gemma-4-31b-it" # Execution & Coding
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = "gemma-4-26b-it"  # Quick background tasks

Write-Host "Launching Claude Code with hybrid opusplan logic..." -ForegroundColor Cyan
claude 