from groq import AsyncGroq
from typing import Dict, Optional, Literal
import os
from dotenv import load_dotenv

load_dotenv()

# Available LLM models and their configurations
LLM_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "description": "LLaMA 4 Scout - High-quality instruction-tuned model for accurate medical reasoning",
        "max_tokens": 8192,
        "default_temp": 0.2
    },
    "meta-llama/llama-guard-4-12b": {
        "description": "LLaMA Guard - Safety-tuned model for filtering and validating medical outputs",
        "max_tokens": 8192,
        "default_temp": 0.1
    },
    "gemma2-9b-it": {
        "description": "Gemma 2 - Lightweight instruction-tuned model for medical summarization and Q&A",
        "max_tokens": 8192,
        "default_temp": 0.3
    }
}


class MedicalGroqService:
    def __init__(self, default_model: str = "gemma2-9b-it"):
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        if default_model not in LLM_MODELS:            raise ValueError(f"Model {default_model} not supported. Available models: {list(LLM_MODELS.keys())}")
        self.current_model = default_model
        self.model_config = LLM_MODELS[default_model]

    def set_model(self, model_name: str) -> None:
        """Change the active LLM model"""
        if model_name not in LLM_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(LLM_MODELS.keys())}")
        self.current_model = model_name
        self.model_config = LLM_MODELS[model_name]

    @property
    def available_models(self) -> Dict:
        """Get information about available models"""
        return LLM_MODELS

    async def generate_medical_prompt(
        self,
        image_type: str,
        prediction_data: Dict,
        patient_context: Optional[Dict] = None,
        summary_mode: bool = True
    ) -> tuple[str, str]:
        """Generate a structured medical prompt for the LLM"""

        system_prompt = """You are a medical AI assistant. You do NOT have access to the actual medical image.
Your role is to interpret the output from an AI model that has already analyzed the image.
Use the predicted class, confidence, and optional patient context to generate a medically sound interpretation.
Do NOT speculate on imaging appearance or features.
Be medically accurate, concise (under 300 words), and clear for healthcare professionals."""

        confidence = prediction_data.get("confidence", 0)
        caution_note = ""
        if confidence < 50:
            caution_note = "⚠️ The AI confidence is low. Reflect uncertainty and recommend further validation."

        if summary_mode:
            focus_note = "Please keep your answer concise, clinically focused, and under 250 words."
        else:
            focus_note = "You may include more detailed guidance (max 300 words)."

        user_prompt = f"""
An AI model analyzed a {image_type} image and returned the following results:

- **Predicted Class**: {prediction_data.get('predicted_class', 'Unknown')}
- **AI Confidence**: {confidence}%
- **Model Output**: {prediction_data.get('diagnosis', 'No diagnosis provided')}

Patient Context:
{patient_context if patient_context else 'No additional patient context provided'}

{caution_note}

{focus_note}

Please return a structured interpretation including:

1. Interpretation of the AI result
2. Clinical significance and possible diagnoses
3. Next diagnostic or follow-up steps
4. Relevant referrals if needed
5. Caveats or warnings (especially if low confidence)
"""
        return system_prompt, user_prompt

    async def analyze_medical_image(
        self,
        image_type: str,
        prediction_data: Dict,
        patient_context: Optional[Dict] = None,
        model_params: Optional[Dict] = None,
        summary_mode: bool = True
    ) -> Dict:
        """Perform a medical image interpretation using selected Groq LLM"""

        try:
            system_prompt, user_prompt = await self.generate_medical_prompt(
                image_type,
                prediction_data,
                patient_context,
                summary_mode=summary_mode
            )

            params = {
                "temperature": self.model_config["default_temp"],
                "max_tokens": self.model_config["max_tokens"]
            }
            if model_params:
                params.update(model_params)

            response = await self.client.chat.completions.create(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **params
            )

            return {
                "analysis": response.choices[0].message.content.strip(),
                "model_info": {
                    "model": self.current_model,
                    "temperature": params["temperature"],
                    "max_tokens": params["max_tokens"],
                    "type": "medical_analysis"
                }
            }

        except Exception as e:
            return {
                "error": f"LLM analysis failed: {str(e)}",
                "analysis": None,
                "model_info": None
            }

    async def get_model_info(self) -> Dict:
        """Get current model configuration"""
        return {
            "current_model": self.current_model,
            "configuration": self.model_config,
            "available_models": self.available_models
        }