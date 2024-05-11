# Get the directory of the script
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define the virtual environment directory
$venvDir = Join-Path -Path $scriptDir -ChildPath "venv"

# Check if Python 3.9.13 is installed
$pythonVersion = "3.9.13"
$pythonExe = "python$pythonVersion"
if (!(Test-Path (Join-Path $env:ProgramFiles "$pythonExe\python.exe"))) {
    Write-Host "Python $pythonVersion not found. Downloading and installing..."

    # Download Python 3.9.13 installer
    Invoke-WebRequest "https://www.python.org/ftp/python/$pythonVersion/python-$pythonVersion-amd64.exe" -OutFile "$scriptDir\python-$pythonVersion-amd64.exe"

    # Install Python 3.9.13
    Start-Process -Wait -FilePath "$scriptDir\python-$pythonVersion-amd64.exe" -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1"
}

# Create the virtual environment
python -m venv $venvDir
Write-Host "Virtual environment created at: $venvDir"

# Activate the virtual environment
. "$venvDir\Scripts\Activate"

# Install required packages using pip
pip install -q pyyaml numpy matplotlib tensorflow

# Deactivate the virtual environment
deactivate

Write-Host "Packages installed successfully."
