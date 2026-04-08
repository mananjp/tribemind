Write-Host "Installing dependencies... This will take a few minutes."
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn pydantic huggingface_hub transformers accelerate "numpy<2.0.0" scipy nibabel nilearn opencv-python-headless git+https://github.com/facebookresearch/tribev2.git
Write-Host ""
Write-Host "Done! You can now run the server with:"
Write-Host ".\.venv\Scripts\python.exe server.py"
