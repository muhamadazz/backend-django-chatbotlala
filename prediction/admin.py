from django.contrib import admin
from .models import PredictionHistory

@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['predicted_label', 'confidence_score', 'timestamp']
    list_filter = ['predicted_label', 'timestamp']
    search_fields = ['text_input', 'predicted_label']
    readonly_fields = ['timestamp']
    ordering = ['-timestamp']