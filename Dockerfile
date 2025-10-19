# Use a small official Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Expose Streamlit's port
EXPOSE 8501

# Run Streamlit when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
