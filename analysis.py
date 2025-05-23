from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from datetime import datetime
from typing import Dict, Optional
import os
from dotenv import load_dotenv
from llm_service import MedicalGroqService

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Medical Image Analysis API",
    description="Advanced medical image analysis with LLM-enhanced reporting",
    version="2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants and configurations
MODEL_PATHS = {
    "melanoma": "models/melanoma_detector_modelvf.h5",
    "brain": "models/modelBrainTumor.h5",
    "pneumonia": "models/pneumonie_model.h5"
}

CLASS_NAMES = {
    "melanoma": ['Melanoma', 'NotMelanoma'],
    "brain": ['No Tumor', 'Tumor Present'],
    "pneumonia": ['NORMAL', 'PNEUMONIA']
}

MEDICAL_CONTEXTS = {
    "melanoma": {
        "description": "Skin lesion analysis for melanoma detection",
        "risk_factors": ["UV exposure", "Family history", "Previous melanoma"],
        "follow_up_tests": ["Dermoscopy", "Biopsy", "Lymph node examination"],
        "image_type": "dermatological"
    },
    "brain": {
        "description": "Brain MRI analysis for tumor detection",
        "risk_factors": ["Headaches", "Neurological symptoms", "Family history"],
        "follow_up_tests": ["Contrast-enhanced MRI", "Biopsy", "PET scan"],
        "image_type": "neurological"
    },
    "pneumonia": {
        "description": "Chest X-ray analysis for pneumonia detection",
        "risk_factors": ["Respiratory symptoms", "Fever", "Compromised immunity"],
        "follow_up_tests": ["Blood tests", "Sputum culture", "CT scan"],
        "image_type": "radiological"
    }
}

# Initialize services
llm_service = MedicalGroqService()
models = {}

# Load ML models
try:
    for key, path in MODEL_PATHS.items():
        models[key] = tf.keras.models.load_model(path)
        print(f"Successfully loaded {key} model")
        
    # Verify model outputs shape
    def verify_model_outputs():
        for key, model in models.items():
            # Create a dummy input of the right shape
            dummy_input = np.zeros((1, 224, 224, 3))
            try:
                result = model.predict(dummy_input)
                print(f"Model {key} output shape: {result.shape}, output range: [{np.min(result)}, {np.max(result)}]")
            except Exception as e:
                print(f"Model {key} verification failed: {e}")

    verify_model_outputs()
except Exception as e:
    print(f"Error loading models: {e}")

async def preprocess_image(image_file: UploadFile, condition_type: str, target_size=(224, 224)):
    """Preprocess uploaded image for model prediction with model-specific processing"""
    try:
        contents = await image_file.read()
        img = Image.open(io.BytesIO(contents))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
          # Model-specific preprocessing
        if condition_type == "melanoma":
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        elif condition_type == "pneumonia":
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        elif condition_type == "brain":
            # Brain tumor model uses simple scaling to [0,1] range
            img_array = img_array / 255.0
        else: 
            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
            
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

@app.post("/predict/{condition_type}")
async def predict(
    condition_type: str,
    file: UploadFile = File(...),
    patient_data: Optional[Dict] = None,
    summary_mode: bool = True
):
    """Enhanced prediction endpoint with LLM analysis"""
    if condition_type not in models:
        raise HTTPException(status_code=400, detail=f"Model type {condition_type} not supported")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # ML Model prediction with model-specific preprocessing
        img_array = await preprocess_image(file, condition_type)
        prediction = models[condition_type].predict(img_array)
        
        # Model-specific prediction logic
        if condition_type == "melanoma":
            # Melanoma model expects output in categorical format
            predicted_class = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class])
        elif condition_type == "pneumonia":
            # Pneumonia model has a single output neuron with sigmoid
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])
        elif condition_type == "brain":
            # Brain tumor model has a single output neuron with sigmoid
            # Based on the original Flask implementation
            prediction_value = prediction[0][0]  # Extract the single prediction value
            predicted_class = 1 if prediction_value > 0.5 else 0
            confidence = float(prediction_value if predicted_class == 1 else 1 - prediction_value)
        else:
            # Default approach
            predicted_class = 1 if prediction[0][0] > 0.5 else 0
            confidence = float(prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0])
        
        prediction_data = {
            "predicted_class": CLASS_NAMES[condition_type][predicted_class],
            "confidence": round(confidence * 100, 2),
            "diagnosis": f"High confidence {CLASS_NAMES[condition_type][predicted_class]}" 
                       if confidence > 0.75 else f"Possible {CLASS_NAMES[condition_type][predicted_class]}",
            "probability": float(confidence * 100)
        }
          # Get enhanced LLM analysis
        llm_result = await llm_service.analyze_medical_image(
            condition_type,
            prediction_data,
            patient_data,
            summary_mode=summary_mode
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "request_id": os.urandom(8).hex(),
            "condition_type": condition_type,
            "ml_prediction": prediction_data,
            "llm_analysis": llm_result['analysis'],
            "medical_context": MEDICAL_CONTEXTS[condition_type],
            "model_info": {
                "ml_model": f"{condition_type}_detector",
                "llm_model": llm_result.get('model_info', {})
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail={"error": str(e), "type": "prediction_error"}
        )

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "api_name": "Medical Image Analysis API",
        "version": "2.0",
        "models_available": CLASS_NAMES,
        "endpoints": {
            "/predict/{condition_type}": {
                "supported_types": list(MODEL_PATHS.keys()),
                "methods": ["POST"],
                "description": "Analyze medical images with ML and LLM enhancement"
            }
        }
    }