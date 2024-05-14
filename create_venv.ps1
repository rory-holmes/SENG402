# Install required packages using pip
conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11" 
# Deactivate the virtual environment
conda deactivate tf

Write-Host "Packages installed successfully."
