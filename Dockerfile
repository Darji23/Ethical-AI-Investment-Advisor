# Dockerfile for Ethical AI Investment Advisor (Hugging Face Space Compatible)
FROM python:3.11-slim

WORKDIR /code

# Install system dependencies for compiling wheels if needed
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies and install
COPY requirements.txt .
# CPU-only torch keeps the image small (the local Hugging Face LLM tier runs on CPU here)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Set up storage directories with correct access permissions
RUN mkdir -p data logs && chmod -R 777 data logs

# Hugging Face Spaces expect exposed port 7860
EXPOSE 7860

# Run FastAPI server, defaulting to port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
