# ---------- 1. CUDA base image with Python 3.10 ---------- #
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ---------- 2. System libs for OpenCV & FFmpeg ---------- #
# Install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

    # 2.5. Create symlink: make "python" available
RUN ln -s /usr/bin/python3 /usr/bin/python
# ---------- 3. Python deps ---------- #
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 4. Copy source code ---------- #
COPY . .

# editable install of the sam2 library inside sam2repo
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ---------- 5. Expose Flask port & start ---------- #
ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "app.py"]

