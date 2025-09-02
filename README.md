# Mental Health Prediction API

A Django REST Framework API that uses an LSTM model to detect mental health conditions based on text input.

## Features

- **Text-based Mental Health Prediction**: Analyzes text input to predict mental health categories
- **RESTful API**: Clean and well-documented API endpoints
- **Prediction History**: Stores and retrieves prediction history
- **Health Check**: Monitor API status and model loading
- **Admin Interface**: Django admin for managing prediction data

## API Endpoints

### 1. Make Prediction
**POST** `/api/predict/`

Analyzes text input and returns mental health prediction.

**Request Body:**
```json
{
  "text": "I feel really sad and hopeless lately. Nothing seems to matter anymore."
}
```

**Response:**
```json
{
  "predicted_label": "depression",
  "confidence_score": 0.87,
  "text_input": "I feel really sad and hopeless lately. Nothing seems to matter anymore."
}
```

### 2. Get Prediction History
**GET** `/api/history/`

Retrieves the last 50 predictions made by the API.

**Response:**
```json
[
  {
    "id": 1,
    "text_input": "I feel really sad...",
    "predicted_label": "depression",
    "confidence_score": 0.87,
    "timestamp": "2024-01-15T10:30:00Z"
  }
]
```

### 3. Health Check
**GET** `/api/health/`

Checks if the API and model are working properly.

**Response:**
```json
{
  "status": "healthy",
  "message": "Mental Health Prediction API is running",
  "model_loaded": true
}
```

## Model Files Required

The following model files should be placed in the project root directory:

- `subreddit_lstm_model.keras` or `subreddit_lstm_model.h5` - The trained LSTM model
- `tokenizer.pkl` - Text tokenizer for preprocessing
- `subreddit_label_encoder.pkl` - Label encoder for output classes

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd mental-health-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run migrations**
```bash
python manage.py migrate
```

5. **Create superuser (optional)**
```bash
python manage.py createsuperuser
```

6. **Start development server**
```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/`

## Deployment

### Heroku Deployment

1. **Install Heroku CLI**
2. **Login to Heroku**
```bash
heroku login
```

3. **Create Heroku app**
```bash
heroku create your-app-name
```

4. **Set environment variables**
```bash
heroku config:set SECRET_KEY="your-secret-key"
heroku config:set DEBUG=False
```

5. **Deploy**
```bash
git add .
git commit -m "Initial deployment"
git push heroku main
```

### Environment Variables

For production deployment, set these environment variables:

- `SECRET_KEY`: Django secret key (required)
- `DEBUG`: Set to `False` for production
- `ALLOWED_HOSTS`: Comma-separated list of allowed hosts

## Usage Examples

### Using curl

```bash
# Make a prediction
curl -X POST http://localhost:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"text": "I have been feeling anxious and worried about everything lately"}'

# Check health
curl http://localhost:8000/api/health/

# Get history
curl http://localhost:8000/api/history/
```

### Using Python requests

```python
import requests

# Make prediction
response = requests.post(
    'http://localhost:8000/api/predict/',
    json={'text': 'I feel overwhelmed and stressed all the time'}
)
result = response.json()
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence_score']:.2f}")
```

### Using JavaScript fetch

```javascript
// Make prediction
const response = await fetch('http://localhost:8000/api/predict/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'I have trouble sleeping and concentrating'
  })
});

const result = await response.json();
console.log('Prediction:', result.predicted_label);
console.log('Confidence:', result.confidence_score);
```

## Model Information

This API uses an LSTM (Long Short-Term Memory) neural network trained to classify mental health conditions based on text patterns. The model:

- Processes text input through tokenization and sequence padding
- Uses pre-trained embeddings for text representation
- Outputs probability distributions across different mental health categories
- Provides confidence scores for each prediction

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input format or missing required fields
- **500 Internal Server Error**: Model loading errors or prediction failures
- **404 Not Found**: Invalid endpoints

## Security Considerations

- CORS is enabled for development (configure appropriately for production)
- Input validation prevents malicious text injection
- Model files should be secured and not publicly accessible
- Consider rate limiting for production use

## Monitoring and Logging

- All predictions are logged for monitoring
- Prediction history is stored in the database
- Health check endpoint for monitoring API status
- Error logging for debugging and maintenance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.