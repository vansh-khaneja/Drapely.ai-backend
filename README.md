# Drapely AI Virtual Try-On Microservice

A FastAPI microservice for virtual try-on functionality in fashion applications.

## Features

- **Trial Endpoint**: Process 2-3 images for virtual try-on (limited features)
- **Premium Endpoint**: Process multiple images with advanced features
- RESTful API design
- Request validation and error handling

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check
- `GET /health` - Check service health

### Trial Virtual Try-On
- `POST /api/v1/trial`
- **Limits**: 
  - Maximum 3 total images
  - Maximum 2 garment images
- **Request Body**:
```json
{
  "user_id": "user123",
  "garment_images": ["url1", "url2"],
  "person_image": "url3" // optional
}
```

### Premium Virtual Try-On
- `POST /api/v1/premium`
- **Limits**:
  - Maximum 50 total images
  - Maximum 20 garment images
- **Request Body**:
```json
{
  "user_id": "user123",
  "garment_images": ["url1", "url2", ...],
  "person_image": "url3", // optional
  "options": {} // optional premium options
}
```

## Example Usage

### Trial Endpoint
```bash
curl -X POST "http://localhost:8000/api/v1/trial" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "garment_images": ["https://example.com/garment1.jpg", "https://example.com/garment2.jpg"]
  }'
```

### Premium Endpoint
```bash
curl -X POST "http://localhost:8000/api/v1/premium" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "garment_images": ["https://example.com/garment1.jpg", "https://example.com/garment2.jpg"],
    "person_image": "https://example.com/person.jpg"
  }'
```

## Next Steps

- Implement actual virtual try-on processing logic
- Add image upload support (currently accepts URLs/base64)
- Add database integration for user tracking
- Add authentication/authorization
- Add rate limiting
- Add image storage service integration

