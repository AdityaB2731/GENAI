FROM python:3.13.5-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the app and model files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Healthcheck for Hugging Face Spaces
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
