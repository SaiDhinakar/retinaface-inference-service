# RetinaFace Inference Service

A production-grade face detection inference service using RetinaFace, FastAPI, and SQLAlchemy. Provides REST API endpoints for face detection with bounding boxes, confidence scores, and facial landmarks.

## Features

- **Face Detection**: Detect faces in images with high accuracy using RetinaFace
- **REST API**: FastAPI-based endpoints for image upload and byte-based detection
- **Database Logging**: Store inference metrics (timing, face count, backend used)
- **Multiple Formats**: Support for image files and raw byte data
- **Annotated Output**: Return base64-encoded images with drawn bounding boxes
- **Async Support**: Asynchronous processing for better performance
- **Configurable Database**: SQLite for development, MySQL/PostgreSQL for production

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SaiDhinakar/retinaface-inference-service.git
cd retinaface-inference-service
```

2. Install dependencies:

```bash
uv sync
```

## Configuration

1. Copy the environment file:

```bash
cp .env.example .env
```

2. Edit `.env` with your database configuration:

For development (SQLite - default):

```env
# No DATABASE_URL needed - uses SQLite automatically
```

For production (MySQL):

```env
DATABASE_URL=mysql+aiomysql://user:password@localhost/retinaface
```

For production (PostgreSQL):

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/retinaface
```

## Database Setup

Initialize the database and create tables:

```bash
uv run python database/init_db.py
```

This creates the `face_runtime_metrics` table for logging inference statistics.

## Running the Application

Start the FastAPI server:

```bash
uv run fastapi run api/main.py
```

The API will be available at `http://localhost:8000`

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status": "ok"}
```

## API Usage

### Detect Faces from Image File

**Endpoint**: `POST /detect-from-image`

Upload an image file to detect faces.

**Request**:

```bash
curl -X POST "http://localhost:8000/detect-from-image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

**Response**:

```json
{
  "annotated_image": "base64-encoded-jpeg-image",
  "faces": [
    {
      "bbox": {
        "x1": 100.0,
        "y1": 50.0,
        "x2": 200.0,
        "y2": 150.0
      },
      "score": 0.95,
      "landmarks": [120.5, 80.2, 140.1, 85.3, 130.0, 100.0, 125.5, 115.2, 135.5, 115.8]
    }
  ]
}
```

### Detect Faces from Image Bytes

**Endpoint**: `POST /detect-from-img-bytes`

Send raw image bytes for detection.

**Request** (Python example):

```python
import requests
import cv2

# Load and encode image
img = cv2.imread("path/to/image.jpg")
_, img_encoded = cv2.imencode('.jpg', img)
img_bytes = img_encoded.tobytes()

response = requests.post("http://localhost:8000/detect-from-img-bytes", data=img_bytes)
print(response.json())
```

**Response**: Same format as above.

## Response Schema

- `annotated_image`: Base64-encoded JPEG image with bounding boxes drawn
- `faces`: Array of detected faces
  - `bbox`: Bounding box coordinates (x1, y1, x2, y2)
  - `score`: Confidence score (0-1)
  - `landmarks`: Array of 10 floats representing 5 facial landmarks (x,y pairs)

## Testing

Run the test suite:

```bash
cd tests/
uv run python test_detections.py
```

This tests the `/detect-from-img-bytes` endpoint with a sample image.

## Project Structure

```
retinaface-inference-service/
├── api/
│   ├── main.py          # FastAPI application and endpoints
│   └── schema.py        # Pydantic models
├── database/
│   ├── database.py      # Database connection and session
│   ├── init_db.py       # Database initialization
│   ├── models.py        # SQLAlchemy models
│   └── repository.py    # Database operations
├── runtime/
│   ├── detector.py      # Face detection logic
│   └── process.py       # Image preprocessing/postprocessing
├── tests/
│   └── test_detections.py # API tests
├── utils/
│   └── device.py        # Device detection (CPU/GPU)
├── models/              # Pre-trained model files
├── test_images/         # Test images
├── pyproject.toml       # Project configuration
├── requirements.txt     # Exported dependencies
├── .env.example         # Environment variables template
└── README.md           # This file
```

## Development

### Adding New Features

1. Update API schemas in `api/schema.py`
2. Implement logic in appropriate modules
3. Add database models if needed
4. Update tests
5. Update this README

### Database Migrations

For schema changes, update `database/models.py` and recreate the database:

```bash
rm retinaface.db  # For SQLite
uv run python database/init_db.py
```

## Performance Notes

- Images are resized to max 640px dimension for inference
- Small faces (< 20px) are filtered out
- Inference metrics are logged to database for monitoring
- Supports both CPU and GPU backends (auto-detected)

## Troubleshooting

### Common Issues

1. **Database connection errors**: Check `DATABASE_URL` in `.env`
2. **CUDA errors**: Ensure TensorFlow GPU setup if using GPU
3. **Image loading failures**: Ensure images are valid JPEG/PNG files
4. **Large response times**: Check image resolution and face count

### Logs

Check application logs for detailed error information. Database errors will be logged during metric saving.
