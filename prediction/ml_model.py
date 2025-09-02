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
        self._models_loaded = False
        self._load_error = None
    
    def _lazy_load_models(self):
        """Lazy load the trained LSTM model and preprocessing components"""
        if self._models_loaded:
            return True
            
        if self._load_error:
            raise self._load_error
            
        try:
            # Ubah path model ke folder models di root project
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
            from pathlib import Path
            model_path = Path(model_path)

            # Check if models directory exists
            if not model_path.exists():
                raise FileNotFoundError(f"Models directory not found at {model_path}")
            
            # Load the LSTM model
            model_file = model_path / 'subreddit_lstm_model.keras'
            if model_file.exists():
                logger.info(f"Attempting to load model from: {model_file}")
                self.model = load_model(str(model_file))
                logger.info("LSTM model loaded successfully from .keras file")
            else:
                # Fallback to .h5 file
                model_file = model_path / 'subreddit_lstm_model.h5'
                if model_file.exists():
                    logger.info(f"Attempting to load model from: {model_file}")
                    self.model = load_model(str(model_file))
                    logger.info("LSTM model loaded successfully from .h5 file")
                else:
                    raise FileNotFoundError("No model file found (.keras or .h5)")
            
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
            
            self._models_loaded = True
            return True
                
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            logger.error(error_msg)
            print(error_msg)  # Tambahkan ini untuk debugging di console
            self._load_error = Exception(error_msg)
            raise self._load_error
    
    def preprocess_text(self, text):
        """Preprocess text for model prediction"""
        try:
            # Ensure models are loaded
            self._lazy_load_models()
            
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
            # Ensure models are loaded
            self._lazy_load_models()
            
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
    
    def is_ready(self):
        """Check if the predictor is ready to make predictions"""
        try:
            self._lazy_load_models()
            return True
        except Exception:
            return False

# Global instance - models will be loaded lazily when first used
predictor = MentalHealthPredictor()