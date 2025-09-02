import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class MentalHealthPredictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.max_length = 100  # Adjust based on your model training
        self.load_models()
    
    def load_models(self):
        """Load the trained LSTM model and preprocessing components"""
        try:
            model_path = settings.BASE_DIR / 'models'
            
            # Load the LSTM model
            model_file = model_path / 'subreddit_lstm_model.keras'
            if model_file.exists():
                self.model = load_model(str(model_file))
                logger.info("LSTM model loaded successfully from .keras file")
            else:
                # Fallback to .h5 file
                model_file = model_path / 'subreddit_lstm_model.h5'
                if model_file.exists():
                    self.model = load_model(str(model_file))
                    logger.info("LSTM model loaded successfully from .h5 file")
                else:
                    raise FileNotFoundError("No model file found")
            
            # Load tokenizer
            tokenizer_file = model_path / 'tokenizer.pkl'
            if tokenizer_file.exists():
                with open(tokenizer_file, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Tokenizer loaded successfully")
            else:
                raise FileNotFoundError("Tokenizer file not found")
            
            # Load label encoder
            encoder_file = model_path / 'subreddit_label_encoder.pkl'
            if encoder_file.exists():
                with open(encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully")
            else:
                raise FileNotFoundError("Label encoder file not found")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess text for model prediction"""
        try:
            # Convert text to sequences using the tokenizer
            sequences = self.tokenizer.texts_to_sequences([text])
            
            # Pad sequences to match model input shape
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
            
            return padded_sequences
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    def predict(self, text):
        """Make prediction on input text"""
        try:
            if not all([self.model, self.tokenizer, self.label_encoder]):
                raise ValueError("Models not properly loaded")
            
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Make prediction
            prediction_probs = self.model.predict(processed_text, verbose=0)
            
            # Get the predicted class
            predicted_class_idx = np.argmax(prediction_probs[0])
            confidence_score = float(np.max(prediction_probs[0]))
            
            # Decode the predicted label
            predicted_label = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            return {
                'predicted_label': predicted_label,
                'confidence_score': confidence_score,
                'text_input': text
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise

# Global instance
predictor = MentalHealthPredictor()