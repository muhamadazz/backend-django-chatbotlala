from rest_framework import serializers
from .models import PredictionHistory

class PredictionRequestSerializer(serializers.Serializer):
    text = serializers.CharField(max_length=5000, help_text="Text input for mental health prediction")

class PredictionResponseSerializer(serializers.Serializer):
    predicted_label = serializers.CharField(help_text="Predicted mental health category")
    confidence_score = serializers.FloatField(help_text="Confidence score of the prediction (0-1)")
    text_input = serializers.CharField(help_text="Original text input")

class PredictionHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionHistory
        fields = ['id', 'text_input', 'predicted_label', 'confidence_score', 'timestamp']
        read_only_fields = ['id', 'timestamp']