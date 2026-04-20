FROM python:3.10-bookworm

WORKDIR /app

# Install FFmpeg and the missing build tools
RUN apt-get update && apt-get install -y ffmpeg pkg-config build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip to ensure it downloads pre-compiled wheels properly
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]