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
# Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install Python dependencies
# -----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Create folders used at runtime
# -----------------------------
RUN mkdir -p models outputs

# -----------------------------
# Expose port (Render ใช้ 10000)
# -----------------------------
EXPOSE 10000

# -----------------------------
# Start FastAPI
# -----------------------------
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
