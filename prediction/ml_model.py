import os
import pickle
import numpy as np
from pathlib import Path
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
            # Import TensorFlow here to avoid import errors at module level
            try:
                import tensorflow as tf
                from tensorflow import keras
                from tensorflow.keras.models import load_model
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                print(f"DEBUG: TensorFlow version: {tf.__version__}")
                print(f"DEBUG: Keras version: {keras.__version__}")
            except ImportError as e:
                error_msg = f"TensorFlow import failed: {str(e)}"
                print(f"DEBUG: {error_msg}")
                raise ImportError(error_msg)
            
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
            
            # Load label encoder first (simplest component)
            encoder_file = model_path / 'subreddit_label_encoder.pkl'
            if encoder_file.exists():
                logger.info(f"Loading label encoder from: {encoder_file}")
                print(f"DEBUG: Loading label encoder from: {encoder_file}")
                try:
                    with open(encoder_file, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    logger.info("Label encoder loaded successfully")
                    print("DEBUG: Label encoder loaded successfully")
                    print(f"DEBUG: Label encoder classes: {self.label_encoder.classes_}")
                except Exception as e:
                    error_msg = f"Error loading label encoder: {str(e)}"
                    logger.error(error_msg)
                    print(f"DEBUG: {error_msg}")
                    raise Exception(error_msg)
            else:
                error_msg = f"Label encoder file not found at {encoder_file}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Load tokenizer with compatibility handling
            tokenizer_file = model_path / 'tokenizer.pkl'
            if tokenizer_file.exists():
                logger.info(f"Loading tokenizer from: {tokenizer_file}")
                print(f"DEBUG: Loading tokenizer from: {tokenizer_file}")
                try:
                    # Try loading with different methods
                    with open(tokenizer_file, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                    logger.info("Tokenizer loaded successfully")
                    print("DEBUG: Tokenizer loaded successfully")
                    print(f"DEBUG: Tokenizer type: {type(self.tokenizer)}")
                    if hasattr(self.tokenizer, 'word_index'):
                        print(f"DEBUG: Tokenizer word_index size: {len(self.tokenizer.word_index)}")
                except Exception as e:
                    error_msg = f"Error loading tokenizer: {str(e)}"
                    logger.error(error_msg)
                    print(f"DEBUG: {error_msg}")
                    raise Exception(error_msg)
            else:
                error_msg = f"Tokenizer file not found at {tokenizer_file}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Load the LSTM model with multiple fallback methods
            model_file = model_path / 'subreddit_lstm_model.keras'
            if model_file.exists():
                logger.info(f"Loading model from: {model_file}")
                print(f"DEBUG: Loading model from: {model_file}")
                print(f"DEBUG: Model file size: {model_file.stat().st_size} bytes")
                
                try:
                    # Method 1: Try loading with compile=False first (most compatible)
                    print("DEBUG: Trying to load with compile=False...")
                    self.model = load_model(str(model_file), compile=False)
                    logger.info("LSTM model loaded successfully with compile=False")
                    print("DEBUG: LSTM model loaded successfully with compile=False")
                    
                except Exception as e1:
                    print(f"DEBUG: Load with compile=False failed: {str(e1)}")
                    
                    try:
                        # Method 2: Try standard loading
                        print("DEBUG: Trying standard load_model...")
                        self.model = load_model(str(model_file))
                        logger.info("LSTM model loaded successfully")
                        print("DEBUG: LSTM model loaded successfully")
                        
                    except Exception as e2:
                        print(f"DEBUG: Standard load_model failed: {str(e2)}")
                        
                        try:
                            # Method 3: Try loading as .h5 if available
                            h5_file = model_path / 'subreddit_lstm_model.h5'
                            if h5_file.exists():
                                print(f"DEBUG: Trying .h5 file: {h5_file}")
                                self.model = load_model(str(h5_file))
                                logger.info("LSTM model loaded from .h5 file")
                                print("DEBUG: LSTM model loaded from .h5 file")
                            else:
                                error_msg = f"All model loading methods failed. Errors: compile=False={str(e1)}, standard={str(e2)}"
                                logger.error(error_msg)
                                print(f"DEBUG: {error_msg}")
                                raise Exception(error_msg)
                                
                        except Exception as e3:
                            error_msg = f"All model loading methods failed. Final error: {str(e3)}"
                            logger.error(error_msg)
                            print(f"DEBUG: {error_msg}")
                            raise Exception(error_msg)
            else:
                error_msg = f"Model file not found at {model_file}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Verify all components are loaded
            if not all([self.model, self.tokenizer, self.label_encoder]):
                error_msg = "Not all model components were loaded successfully"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise ValueError(error_msg)
            
            # Test the model with a simple prediction to ensure it works
            try:
                print("DEBUG: Testing model with sample input...")
                test_sequences = self.tokenizer.texts_to_sequences(["test input"])
                test_padded = pad_sequences(test_sequences, maxlen=self.max_length, padding="post", truncating="post")
                test_prediction = self.model.predict(test_padded, verbose=0)
                print(f"DEBUG: Model test successful, output shape: {test_prediction.shape}")
            except Exception as e:
                error_msg = f"Model test failed: {str(e)}"
                logger.error(error_msg)
                print(f"DEBUG: {error_msg}")
                raise Exception(error_msg)
            
            self._models_loaded = True
            logger.info("All models loaded and tested successfully!")
            print("DEBUG: All models loaded and tested successfully!")
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
            # Import here to avoid module-level import issues
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
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