from django.urls import path
from .views import PredictMentalHealthView, PredictionHistoryView, health_check

urlpatterns = [
    path('predict/', PredictMentalHealthView.as_view(), name='predict_mental_health'),
    path('history/', PredictionHistoryView.as_view(), name='prediction_history'),
    path('health/', health_check, name='health_check'),
]