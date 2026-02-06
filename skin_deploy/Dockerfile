# -----------------------------
# Base image (Python)
# -----------------------------
FROM python:3.11-slim

# -----------------------------
# Environment settings
# -----------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy project files
# -----------------------------
WORKDIR /app
COPY . .

# -----------------------------
# Move into your app folder (สำคัญมาก)
# -----------------------------
WORKDIR /app/skin_deploy

# -----------------------------
# Install Python dependencies
# -----------------------------
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Create folders used at runtime
# -----------------------------
RUN mkdir -p models outputs

# -----------------------------
# Expose port (Hugging Face Spaces ใช้ 7860)
# -----------------------------
EXPOSE 7860

# -----------------------------
# Start FastAPI (Hugging Face Spaces)
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
