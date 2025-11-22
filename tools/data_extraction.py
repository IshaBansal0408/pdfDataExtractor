"""
Data Extraction Module using Ollama LLM
This module processes raw PDF text and extracts structured data using local LLM.
Converts unstructured text into JSON format matching the expected output structure.
"""

import json
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaDataExtractor:
    """
    Data extraction using Ollama local LLM.
    Processes raw PDF text and extracts structured information.
    """

    def __init__(
        self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama Data Extractor.

        Args:
            model (str): Ollama model to use for extraction
            base_url (str): Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.model_name = model  # Will be updated by select_best_model()

        # Comprehensive data structure matching Expected Output (36 fields)
        self.expected_fields = {
            # Personal Information
            "first_name": {"type": "string", "description": "Person's first name"},
            "last_name": {"type": "string", "description": "Person's last name"},
            "date_of_birth": {
                "type": "date",
                "description": "Birth date in YYYY-MM-DD format",
            },
            "birth_city": {
                "type": "string",
                "description": "City where person was born",
            },
            "birth_state": {
                "type": "string",
                "description": "State where person was born",
            },
            "age": {
                "type": "string",
                "description": "Current age with units (e.g., '35 years')",
            },
            "blood_group": {
                "type": "string",
                "description": "Blood group (e.g., 'O+', 'AB-')",
            },
            "nationality": {"type": "string", "description": "Person's nationality"},
            # First Professional Role
            "joining_date_first_role": {
                "type": "date",
                "description": "Start date of first job in YYYY-MM-DD format",
            },
            "designation_first_role": {
                "type": "string",
                "description": "Job title of first professional role",
            },
            "salary_first_role": {
                "type": "number",
                "description": "Salary amount of first job (numeric)",
            },
            "salary_currency_first_role": {
                "type": "string",
                "description": "Currency of first job salary (INR, USD, etc.)",
            },
            # Current Professional Role
            "current_organization": {
                "type": "string",
                "description": "Name of current company/organization",
            },
            "current_joining_date": {
                "type": "date",
                "description": "Start date of current job in YYYY-MM-DD format",
            },
            "current_designation": {
                "type": "string",
                "description": "Current job title/position",
            },
            "current_salary": {
                "type": "number",
                "description": "Current salary amount (numeric)",
            },
            "current_salary_currency": {
                "type": "string",
                "description": "Current salary currency",
            },
            # Previous Professional Role
            "previous_organization": {
                "type": "string",
                "description": "Name of previous company",
            },
            "previous_joining_date": {
                "type": "date",
                "description": "Start date of previous job",
            },
            "previous_end_year": {
                "type": "number",
                "description": "Year when previous job ended",
            },
            "previous_starting_designation": {
                "type": "string",
                "description": "Starting designation at previous job",
            },
            # Educational Background
            "high_school": {
                "type": "string",
                "description": "Name of high school attended",
            },
            "high_school_pass_year": {
                "type": "number",
                "description": "Year of 12th standard completion",
            },
            "high_school_score": {
                "type": "number",
                "description": "12th standard percentage (as decimal, e.g., 0.925 for 92.5%)",
            },
            # Undergraduate Education
            "undergraduate_degree": {
                "type": "string",
                "description": "Undergraduate degree name (e.g., 'B.Tech (Computer Science)')",
            },
            "undergraduate_college": {
                "type": "string",
                "description": "Name of undergraduate college/university",
            },
            "undergraduate_year": {
                "type": "number",
                "description": "Year of undergraduate graduation",
            },
            "undergraduate_cgpa": {
                "type": "number",
                "description": "Undergraduate CGPA (on 10-point scale)",
            },
            # Graduate Education
            "graduation_degree": {
                "type": "string",
                "description": "Graduate degree name (e.g., 'M.Tech (Data Science)')",
            },
            "graduation_college": {
                "type": "string",
                "description": "Name of graduate college/university",
            },
            "graduation_year": {
                "type": "number",
                "description": "Year of graduate degree completion",
            },
            "graduation_cgpa": {
                "type": "number",
                "description": "Graduate CGPA (on 10-point scale)",
            },
            # Professional Certifications
            "certification_1": {
                "type": "string",
                "description": "First professional certification",
            },
            "certification_2": {
                "type": "string",
                "description": "Second professional certification",
            },
            "certification_3": {
                "type": "string",
                "description": "Third professional certification",
            },
            "certification_4": {
                "type": "string",
                "description": "Fourth professional certification",
            },
        }

    def select_best_model(self) -> bool:
        """
        Automatically select the best available model from Ollama.

        Returns:
            bool: True if a model was selected, False if no models available
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]

                # Priority order: llama3.2:3b > llama3.2:1b > tinyllama > any llama3.2 > first available
                if "llama3.2:3b" in available_models:
                    self.model_name = "llama3.2:3b"
                elif "llama3.2:1b" in available_models:
                    self.model_name = "llama3.2:1b"
                elif "tinyllama:latest" in available_models:
                    self.model_name = "tinyllama:latest"
                elif "tinyllama" in available_models:
                    self.model_name = "tinyllama"
                elif any("llama3.2" in model for model in available_models):
                    self.model_name = next(
                        model for model in available_models if "llama3.2" in model
                    )
                elif available_models:
                    self.model_name = available_models[0]  # Use first available model
                else:
                    logger.error("No models available in Ollama")
                    return False

                logger.info(f"Selected Ollama model: {self.model_name}")
                return True
            else:
                logger.error(f"Failed to get models list: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            return False

    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.

        Returns:
            bool: True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            return False

    def check_model_availability(self) -> bool:
        """
        Check if the specified model is available in Ollama.

        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return self.model in available_models
            return False
        except Exception as e:
            logger.error(f"Error checking model availability: {str(e)}")
            return False

    def create_extraction_prompt(self, raw_text: str) -> str:
        """
        Create a structured prompt for LLM to extract data.

        Args:
            raw_text (str): Raw text from PDF

        Returns:
            str: Formatted prompt for LLM
        """
        field_descriptions = []
        for field, info in self.expected_fields.items():
            field_descriptions.append(
                f"- {field}: {info['description']} ({info['type']})"
            )

        prompt = f"""
You are an expert data extraction AI. Analyze the following text and extract ALL available information into a JSON format.

EXTRACT THESE FIELDS (use "" for missing data, be thorough):

PERSONAL INFO:
- first_name, last_name, date_of_birth (YYYY-MM-DD), birth_city, birth_state, age, blood_group, nationality

FIRST JOB:
- joining_date_first_role (YYYY-MM-DD), designation_first_role, salary_first_role (number), salary_currency_first_role

CURRENT JOB:
- current_organization, current_joining_date (YYYY-MM-DD), current_designation, current_salary (number), current_salary_currency

PREVIOUS JOB:
- previous_organization, previous_joining_date (YYYY-MM-DD), previous_end_year (number), previous_starting_designation

EDUCATION:
- high_school, high_school_pass_year (number), high_school_score (decimal 0-1)
- undergraduate_degree, undergraduate_college, undergraduate_year (number), undergraduate_cgpa (number)
- graduation_degree, graduation_college, graduation_year (number), graduation_cgpa (number)

CERTIFICATIONS:
- certification_1, certification_2, certification_3, certification_4

IMPORTANT:
1. Extract dates in YYYY-MM-DD format
2. Convert percentages to decimals (92.5% → 0.925)
3. Extract only numbers for salary/CGPA fields
4. Be thorough - extract every piece of available information
5. Return ONLY valid JSON, no explanations

Text to analyze:
{raw_text}

JSON:
"""
        return prompt

    def query_ollama(self, prompt: str) -> Optional[str]:
        """
        Send prompt to Ollama and get response.

        Args:
            prompt (str): The prompt to send to LLM

        Returns:
            Optional[str]: LLM response or None if failed
        """
        try:
            payload = {
                "model": self.model_name,  # Use the selected model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "top_p": 0.9,
                },
            }

            logger.info(f"Querying Ollama model: {self.model_name}")
            response = requests.post(
                self.api_url, json=payload, timeout=120
            )  # Increased timeout

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            return None

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON response from LLM, handling potential formatting issues.

        Args:
            response (str): Raw response from LLM

        Returns:
            Optional[Dict[str, Any]]: Parsed JSON or None if parsing failed
        """
        try:
            # Try direct JSON parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try to find JSON within the response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)

                logger.error("No valid JSON found in response")
                return None
            except Exception as e:
                logger.error(f"JSON parsing failed: {str(e)}")
                return None

    def validate_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data.

        Args:
            data (Dict[str, Any]): Raw extracted data

        Returns:
            Dict[str, Any]: Validated and cleaned data
        """
        validated = {}

        for field_name, field_info in self.expected_fields.items():
            value = data.get(field_name, "")

            # Clean and validate based on type
            if field_info["type"] == "date":
                # Ensure date format consistency
                if value and len(str(value)) >= 10:
                    # Try to clean date format
                    date_str = str(value)[:10]  # Take first 10 characters
                    validated[field_name] = date_str
                else:
                    validated[field_name] = ""
            else:
                # String fields
                validated[field_name] = str(value).strip() if value else ""

        return validated

    def extract_data_from_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Main extraction method - processes raw text and returns structured data.

        Args:
            raw_text (str): Raw text content from PDF

        Returns:
            Dict[str, Any]: Extraction results with success status and data
        """
        result = {
            "success": False,
            "extracted_data": {},
            "llm_response": "",
            "model_used": self.model,
            "error_message": None,
        }

        try:
            # Check Ollama connection and select best available model
            if not self.select_best_model():
                result["error_message"] = (
                    "Cannot connect to Ollama or no models available. Please ensure Ollama is running and install a model."
                )
                return result

            # Update result to show which model was actually used
            result["model_used"] = self.model_name

            # Create extraction prompt
            prompt = self.create_extraction_prompt(raw_text)

            # Query LLM
            llm_response = self.query_ollama(prompt)
            if not llm_response:
                result["error_message"] = "Failed to get response from LLM"
                return result

            result["llm_response"] = llm_response

            # Parse JSON response
            extracted_json = self.parse_json_response(llm_response)
            if not extracted_json:
                result["error_message"] = "Failed to parse JSON from LLM response"
                return result

            # Validate and clean data
            validated_data = self.validate_extracted_data(extracted_json)
            result["extracted_data"] = validated_data
            result["success"] = True

            logger.info("Successfully extracted structured data from raw text")

        except Exception as e:
            result["error_message"] = f"Data extraction failed: {str(e)}"
            logger.error(f"Data extraction error: {str(e)}")

        return result

    def save_extracted_data(self, data: Dict[str, Any], output_path: str) -> bool:
        """
        Save extracted data to JSON file.

        Args:
            data (Dict[str, Any]): Extracted data to save
            output_path (str): Path to save the JSON file

        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Extracted data saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save extracted data: {str(e)}")
            return False


# Convenience functions
def extract_data_with_ollama(
    raw_text: str, model: str = "llama3.2:3b"
) -> Dict[str, Any]:
    """
    Convenience function to extract data from raw text using Ollama.

    Args:
        raw_text (str): Raw PDF text content
        model (str): Ollama model name to use

    Returns:
        Dict[str, Any]: Extraction results
    """
    extractor = OllamaDataExtractor(model=model)
    return extractor.extract_data_from_text(raw_text)


def setup_ollama_instructions() -> str:
    """
    Return instructions for setting up Ollama.

    Returns:
        str: Setup instructions
    """
    return """
🔧 Ollama Setup Instructions:

1. Install Ollama:
   curl -fsSL https://ollama.com/install.sh | sh

2. Start Ollama service:
   ollama serve

3. Download a model (choose one):
   ollama pull llama3.2:3b      # 2GB - Fast, good for extraction
   ollama pull mistral:7b       # 4GB - Better accuracy
   ollama pull qwen2.5:7b       # 4GB - Good for structured data

4. Test the model:
   ollama run llama3.2:3b

5. Verify API access:
   curl http://localhost:11434/api/tags

Once setup is complete, run your PDF extraction again!
"""
