import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from django.conf import settings
import logging
from pathlib import Path

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
            # Get absolute path to models directory
            base_dir = Path(__file__).resolve().parent.parent
            model_path = base_dir / 'models'
            
            logger.info(f"Looking for models in: {model_path}")
            print(f"DEBUG: Looking for models in: {model_path}")
            
            # Check if models directory exists
            if not model_path.exists():
                error_msg = f"Models directory not found at {model_path}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # List files in models directory for debugging
            model_files = list(model_path.glob('*'))
            logger.info(f"Files in models directory: {model_files}")
            print(f"DEBUG: Files in models directory: {model_files}")
            
            # Load the LSTM model
            model_file = model_path / 'subreddit_lstm_model.keras'
            if model_file.exists():
                logger.info(f"Loading model from: {model_file}")
                print(f"DEBUG: Loading model from: {model_file}")
                self.model = load_model(str(model_file))
                logger.info("LSTM model loaded successfully from .keras file")
                print("DEBUG: LSTM model loaded successfully")
            else:
                # Fallback to .h5 file
                model_file = model_path / 'subreddit_lstm_model.h5'
                if model_file.exists():
                    logger.info(f"Loading model from: {model_file}")
                    print(f"DEBUG: Loading model from: {model_file}")
                    self.model = load_model(str(model_file))
                    logger.info("LSTM model loaded successfully from .h5 file")
                    print("DEBUG: LSTM model loaded successfully from .h5")
                else:
                    error_msg = f"No model file found at {model_path} (.keras or .h5)"
                    logger.error(error_msg)
                    print(f"DEBUG: {error_msg}")
                    raise FileNotFoundError(error_msg)
            
            # Load tokenizer
            tokenizer_file = model_path / 'tokenizer.pkl'
            if tokenizer_file.exists():
                logger.info(f"Loading tokenizer from: {tokenizer_file}")
                print(f"DEBUG: Loading tokenizer from: {tokenizer_file}")
                with open(tokenizer_file, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                logger.info("Tokenizer loaded successfully")
                print("DEBUG: Tokenizer loaded successfully")
            else:
                error_msg = f"Tokenizer file not found at {tokenizer_file}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Load label encoder
            encoder_file = model_path / 'subreddit_label_encoder.pkl'
            if encoder_file.exists():
                logger.info(f"Loading label encoder from: {encoder_file}")
                print(f"DEBUG: Loading label encoder from: {encoder_file}")
                with open(encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info("Label encoder loaded successfully")
                print("DEBUG: Label encoder loaded successfully")
            else:
                error_msg = f"Label encoder file not found at {encoder_file}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Verify all components are loaded
            if not all([self.model, self.tokenizer, self.label_encoder]):
                error_msg = "Not all model components were loaded successfully"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise ValueError(error_msg)
            
            self._models_loaded = True
            logger.info("All models loaded successfully!")
            print("DEBUG: All models loaded successfully!")
            return True
                
        except Exception as e:
            error_msg = f"Error loading models: {str(e)}"
            logger.error(error_msg)
            print(f"DEBUG: {error_msg}")
            self._load_error = Exception(error_msg)
            raise self._load_error
    
    def preprocess_text(self, text):
        """Preprocess text for model prediction"""
        try:
            # Ensure models are loaded
            self._lazy_load_models()
            
            if not self.tokenizer:
                raise ValueError("Tokenizer not loaded")
            
            # Convert text to sequences using the tokenizer
            sequences = self.tokenizer.texts_to_sequences([text])
            
            # Pad sequences to match model input shape
            padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding="post", truncating="post")
            
            return padded_sequences
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            print(f"DEBUG: Error preprocessing text: {str(e)}")
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
            
            logger.info(f"Prediction made: {predicted_label} with confidence {confidence_score}")
            print(f"DEBUG: Prediction made: {predicted_label} with confidence {confidence_score}")
            
            return {
                'predicted_label': predicted_label,
                'confidence_score': confidence_score,
                'text_input': text
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            print(f"DEBUG: Error making prediction: {str(e)}")
            raise
    
    def is_ready(self):
        """Check if the predictor is ready to make predictions"""
        try:
            self._lazy_load_models()
            return True
        except Exception as e:
            logger.error(f"Model readiness check failed: {str(e)}")
            print(f"DEBUG: Model readiness check failed: {str(e)}")
            return False

# Global instance - models will be loaded lazily when first used
predictor = MentalHealthPredictor()