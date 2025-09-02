from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.http import JsonResponse
from .serializers import (
    PredictionRequestSerializer, 
    PredictionResponseSerializer,
    PredictionHistorySerializer
)
from .models import PredictionHistory
from .ml_model import predictor
import logging

logger = logging.getLogger(__name__)

class PredictMentalHealthView(APIView):
    """
    API endpoint for mental health prediction based on text input
    """
    
    def post(self, request):
        """
        Make a mental health prediction based on input text
        """
        try:
            # Validate input
            serializer = PredictionRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {'error': 'Invalid input', 'details': serializer.errors},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            text_input = serializer.validated_data['text']
            
            # Check if models are ready
            if not predictor.is_ready():
                return Response(
                    {'error': 'Model not available', 'message': 'Model files are missing or corrupted'},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Make prediction
            prediction_result = predictor.predict(text_input)
            
            # Save to history
            PredictionHistory.objects.create(
                text_input=text_input,
                predicted_label=prediction_result['predicted_label'],
                confidence_score=prediction_result['confidence_score']
            )
            
            # Return response
            response_serializer = PredictionResponseSerializer(data=prediction_result)
            if response_serializer.is_valid():
                return Response(response_serializer.data, status=status.HTTP_200_OK)
            else:
                return Response(
                    {'error': 'Error formatting response'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return Response(
                {'error': 'Internal server error', 'message': str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class PredictionHistoryView(APIView):
    """
    API endpoint to retrieve prediction history
    """
    
    def get(self, request):
        """
        Get all prediction history
        """
        try:
            predictions = PredictionHistory.objects.all()[:50]  # Limit to last 50
            serializer = PredictionHistorySerializer(predictions, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return Response(
                {'error': 'Error retrieving history'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

@api_view(['GET'])
def health_check(request):
    """
    Simple health check endpoint
    """
    try:
        model_status = predictor.is_ready()
        status_code = 200 if model_status else 503
        
        # Additional debugging info
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / 'models'
        
        debug_info = {
            'models_directory_exists': model_path.exists(),
            'model_path': str(model_path),
            'files_in_models': [f.name for f in model_path.glob('*')] if model_path.exists() else []
        }
        
        response_data = {
            'status': 'healthy' if model_status else 'unhealthy',
            'message': 'Mental Health Prediction API is running',
            'model_loaded': model_status,
            'error': None if model_status else str(predictor._load_error) if predictor._load_error else 'Model files not found or corrupted',
            'debug': debug_info
        }
        
        return JsonResponse(response_data, status=status_code)
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': 'Health check failed',
            'model_loaded': False,
            'error': str(e)
        }, status=503)