from django.db import models

class PredictionHistory(models.Model):
    text_input = models.TextField()
    predicted_label = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"Prediction: {self.predicted_label} ({self.confidence_score:.2f})"