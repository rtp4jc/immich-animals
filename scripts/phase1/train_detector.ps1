# PowerShell wrapper to train YOLOv8 pose model
# Activates conda environment and runs training with logging

param()

# Ensure logs directory exists
$logsDir = "outputs/phase1/logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force
}

Write-Host "Activating conda environment 'python312'..."
& "C:\ProgramData\Miniconda3\Scripts\conda.exe" activate python312

Write-Host "Starting YOLOv8 pose model training..."
Write-Host "Logs will be saved to: $logsDir\train.log"

# Run training script and redirect both stdout and stderr to log file
# Use Tee-Object for real-time output to console and file
python scripts/phase1/train_detector.py 2>&1 | Tee-Object -FilePath "$logsDir\train.log"

Write-Host "Training completed. Check logs at $logsDir\train.log"