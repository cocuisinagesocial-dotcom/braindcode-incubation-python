# Use a lightweight Python image
FROM python:3.10.16

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt .

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port Flask will run on
EXPOSE 5006

#install ollama3
CMD ["ollama", "pull", "llama3"]
# Run Flask app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5006"]