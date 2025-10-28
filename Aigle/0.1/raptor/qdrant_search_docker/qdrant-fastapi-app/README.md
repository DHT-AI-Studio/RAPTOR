# qdrant-fastapi-app

This project is a FastAPI application that provides various endpoints for similarity searches across different media types (audio, documents, images, and videos) using Qdrant as the vector database. It also includes functionality for inserting data into Qdrant and managing caching without using Redis.

## Project Structure

```
qdrant-fastapi-app
├── src
│   ├── api_audio_search_with_cache.py       # FastAPI app for audio similarity search
│   ├── api_document_search_with_cache.py    # FastAPI app for document similarity search
│   ├── api_image_search_with_cache.py       # FastAPI app for image similarity search
│   ├── api_insert.py                         # Functionality to insert data into Qdrant
│   ├── api_video_search_with_cache.py       # FastAPI app for video similarity search
│   ├── cache_manager                         # Directory for cache management
│   │   ├── __init__.py                       # Initializes the cache manager module
│   │   └── [your_cache_manager_files].py     # Implementation of the cache manager
│   └── [other_python_modules].py             # Placeholder for additional Python modules
├── qdrant
│   └── docker-compose.yml                     # Configuration for running Qdrant in Docker
├── Dockerfile                                 # Dockerfile for building the FastAPI application image
├── requirements.txt                           # Python dependencies for the project
└── README.md                                  # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine.

   ```bash
   git clone <repository-url>
   cd qdrant-fastapi-app
   ```

2. **Install Dependencies**: 
   Install the required Python packages listed in `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run Qdrant with Docker**: 
   Use Docker Compose to start the Qdrant service.

   ```bash
   cd qdrant
   docker-compose up -d
   ```

4. **Build the Docker Image**: 
   Build the Docker image for the FastAPI application.

   ```bash
   docker build -t qdrant-fastapi-app .
   ```

5. **Run the FastAPI Application**: 
   You can run the FastAPI application using Docker or directly on your local machine.

   To run with Docker:

   ```bash
   docker run -d -p 8811:8811 qdrant-fastapi-app
   ```

   To run locally (if not using Docker):

   ```bash
   uvicorn src/api_video_search_with_cache:app --host 0.0.0.0 --port 8811 --reload
   ```

## Usage

- Access the FastAPI application at `http://localhost:8811/docs` to view the API documentation and test the endpoints.
- Use the `/video_search`, `/audio_search`, `/document_search`, and `/image_search` endpoints to perform similarity searches.
- Use the `/insert_json` endpoint to insert data into Qdrant.

## Notes

- Ensure that Docker is installed and running on your machine to use the Docker Compose functionality.
- Adjust the `docker-compose.yml` and `Dockerfile` as needed to fit your specific requirements and configurations.