 FROM python:3.10-slim

# Install system dependencies (OCR + PDF rendering)
# RUN apt-get update && apt-get install -y \
#     tesseract-ocr \
#     poppler-utils \
#     libgl1 \
#     && rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app

# Copy app files
COPY . .
# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
