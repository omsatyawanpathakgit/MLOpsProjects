# Use a slim Python image
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Copy only requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre‑computed pickles and the Flask app
COPY dishes_similarity.pkl dishes.pkl app.py ./

# Expose the port used by Flask
EXPOSE 5000

# Start the API server
CMD ["python", "app.py"]