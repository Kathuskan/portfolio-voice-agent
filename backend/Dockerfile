FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy and install Python dependencies first (for faster Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY app.py .

# Create a non-root user (Required by Hugging Face Spaces security policies)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Expose port 7860 (Default port expected by Hugging Face Spaces)
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]