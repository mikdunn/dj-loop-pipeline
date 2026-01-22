# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy repository files
COPY . /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
# Install dependencies from project requirements file
RUN pip install -r requirements_dj_loop_pipeline.txt

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Default command (interactive shell, can be overridden)
CMD ["bash"]
