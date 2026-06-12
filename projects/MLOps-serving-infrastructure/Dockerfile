# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency definition
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code files
COPY main.py schemas.py drift_detector.py ./

# Expose FastAPI default port
EXPOSE 8000

# Command to execute FastAPI server using uvicorn ASGI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
