# MedicalVision-AI

![MedicalVision-AI Logo](https://img.shields.io/badge/MedicalVision-AI-blue?style=for-the-badge)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

## About

MedicalVision-AI is an advanced medical image analysis API that combines deep learning models with large language model (LLM) capabilities to provide comprehensive medical diagnostics for critical conditions:

- **Melanoma Detection** (skin lesion analysis)
- **Brain Tumor Detection** (MRI analysis)
- **Pneumonia Detection** (chest X-ray analysis)

## Key Features

- ✅ **Multi-condition Support**: Three specialized medical models in one API
- ✅ **LLM-Enhanced Analysis**: Combines ML predictions with natural language explanations
- ✅ **Model-Specific Processing**: Custom preprocessing pipelines for each medical condition
- ✅ **Comprehensive Results**: Includes medical context and follow-up recommendations
- ✅ **Fast & Scalable**: Built with FastAPI for high performance and concurrency
- ✅ **Medical Context**: Provides condition-specific risk factors and recommended tests

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/MedicalVision-AI.git
cd MedicalVision-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create models directory and add your trained models
mkdir -p models
# Copy your .h5 model files to the models directory
```

## Environment Setup

Create a `.env` file in the project root:

```
LLM_API_KEY=your_groq_api_key_here
LLM_MODEL=llama3-70b-8192
```

## API Usage

### Starting the API

```bash
uvicorn analysis:app --reload
```

The API will be available at http://localhost:8000

### Interactive Documentation

Access the interactive API documentation at http://localhost:8000/docs

### Making Predictions

```python
import requests

# Example: Melanoma prediction
files = {'file': ('image.jpg', open('path/to/image.jpg', 'rb'), 'image/jpeg')}
patient_data = {'age': 45, 'gender': 'female', 'symptoms': ['irregular mole', 'changing color']}

response = requests.post(
    'http://localhost:8000/predict/melanoma',
    files=files,
    json={'patient_data': patient_data}
)

print(response.json())
```

## API Response Example

```json
{
  "timestamp": "2025-05-23T14:32:45.123456",
  "request_id": "a1b2c3d4e5f6g7h8",
  "condition_type": "melanoma",
  "ml_prediction": {
    "predicted_class": "Melanoma",
    "confidence": 87.54,
    "diagnosis": "High confidence Melanoma",
    "probability": 87.54
  },
  "llm_analysis": "The image shows an irregularly shaped lesion with varying colors...",
  "medical_context": {
    "description": "Skin lesion analysis for melanoma detection",
    "risk_factors": ["UV exposure", "Family history", "Previous melanoma"],
    "follow_up_tests": ["Dermoscopy", "Biopsy", "Lymph node examination"],
    "image_type": "dermatological"
  },
  "model_info": {
    "ml_model": "melanoma_detector",
    "llm_model": {
      "name": "llama3-70b-8192",
      "version": "1.0"
    }
  }
}
```

## Project Structure

```
MedicalVision-AI/
├── analysis.py          # Main API implementation
├── llm_service.py       # LLM integration service
├── models/              # Directory for ML models
│   ├── melanoma_detector_modelvf.h5
│   ├── modelBrainTumor.h5
│   └── pneumonie_model.h5
├── .env                 # Environment variables
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## Technical Details

### Model-Specific Processing

Each medical condition requires unique preprocessing:

- **Melanoma**: Uses EfficientNet preprocessing for dermatological images
- **Brain Tumor**: Uses normalized scaling (0-1) for MRI scans
- **Pneumonia**: Uses VGG16 preprocessing for chest X-rays

### LLM Integration

The API integrates with Groq's LLM API to provide enhanced medical analysis and explanations based on the ML model predictions.

## Disclaimer

This tool is designed for preliminary screening and medical assistance only. It should not replace professional medical diagnosis or consultation with healthcare providers.

## License

MIT License

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

Your Name - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/MedicalVision-AI](https://github.com/yourusername/MedicalVision-AI)