FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY src/ ./src/

# Expose the port the app runs on
EXPOSE 8811

# Command to run the FastAPI application
CMD ["uvicorn", "src/api_video_search_with_cache:app", "--host", "0.0.0.0", "--port", "8811", "--reload"]