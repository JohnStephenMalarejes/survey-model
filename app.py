
import os
import re
import json
import logging
import threading
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from llama_cpp import Llama


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

app = Flask(__name__)

app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 2. ENVIRONMENT VARIABLES
# ============================================================

MODEL_GGUF_PATH = os.getenv(
    "MODEL_GGUF_PATH",
    "./model/ai_survey_qwen_3b_q4.gguf"
)

MODEL_GGUF_PATH = Path(MODEL_GGUF_PATH)

if not MODEL_GGUF_PATH.is_absolute():
    MODEL_GGUF_PATH = PROJECT_DIR / MODEL_GGUF_PATH

N_CTX = int(os.getenv("N_CTX", "1024"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_BATCH = int(os.getenv("N_BATCH", "64"))

DEFAULT_MAX_NEW_TOKENS = int(
    os.getenv("DEFAULT_MAX_NEW_TOKENS", "700")
)

MAX_ALLOWED_NEW_TOKENS = int(
    os.getenv("MAX_ALLOWED_NEW_TOKENS", "900")
)

API_KEY = os.getenv("API_KEY", "").strip()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "5000"))


# ============================================================
# 3. CORS CONFIGURATION
# ============================================================

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000"
    ).split(",")
    if origin.strip()
]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": allowed_origins
        }
    },
    supports_credentials=False
)


# ============================================================
# 4. AUTHENTICATION
# ============================================================

@app.before_request
def authenticate_request():
    """
    Optional API-key authentication.

    If API_KEY is empty in .env, authentication is disabled.
    Otherwise, clients must send X-API-Key.
    """

    if request.method == "OPTIONS":
        return None

    if not request.path.startswith("/api/"):
        return None

    if not API_KEY:
        return None

    supplied_key = request.headers.get("X-API-Key", "")

    import hmac

    if not hmac.compare_digest(supplied_key, API_KEY):
        return jsonify({
            "success": False,
            "error": "Unauthorized. Invalid or missing API key."
        }), 401

    return None


# ============================================================
# 5. SURVEY GENERATION SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert AI Survey Generator.

Your job is to create clear, professional, unbiased,
well-structured surveys based on the user's requirements.

RULES:

1. Generate questions relevant to the survey topic.
2. Use simple, understandable language.
3. Avoid leading, biased, or confusing questions.
4. Do not repeat questions.
5. Use appropriate question types.
6. Include answer options for multiple-choice questions.
7. Use an empty options array for open-ended questions.
8. Respect the requested number of questions.
9. Make questions suitable for the specified audience.
10. Return valid JSON only.
11. Do not include Markdown code fences.
12. Do not include explanations outside the JSON.

SUPPORTED QUESTION TYPES:

- multiple_choice
- checkbox
- rating
- scale
- yes_no
- short_answer
- paragraph
- dropdown

REQUIRED JSON STRUCTURE:

{
    "title": "Survey title",
    "description": "Survey description",
    "topic": "Survey topic",
    "target_audience": "Target audience",
    "questions": [
        {
            "id": 1,
            "question": "Question text",
            "type": "multiple_choice",
            "required": true,
            "options": [
                "Option A",
                "Option B",
                "Option C"
            ]
        }
    ]
}

For questions that do not need answer choices,
return an empty options array.

Return only the JSON object.
"""


# ============================================================
# 6. MODEL SERVICE
# ============================================================

class ModelService:

    def __init__(self):
        self.model = None
        self.loaded = False

        self.load_lock = threading.Lock()
        self.generate_lock = threading.Lock()

        self.device = "cpu"

    def load(self):
        """
        Load the quantized Qwen2.5-3B GGUF model.

        The model file must already contain the merged LoRA
        adapter and must be converted to Q4_K_M GGUF.
        """

        if self.loaded:
            return

        with self.load_lock:

            if self.loaded:
                return

            logger.info("Loading AI Survey Generator model...")
            logger.info("Model path: %s", MODEL_GGUF_PATH)

            if not MODEL_GGUF_PATH.is_file():
                raise FileNotFoundError(
                    f"GGUF model not found: {MODEL_GGUF_PATH}. "
                    "Place your quantized model in the model folder "
                    "or update MODEL_GGUF_PATH in your .env file."
                )

            self.model = Llama(
                model_path=str(MODEL_GGUF_PATH),

                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_batch=N_BATCH,

                n_gpu_layers=0,

                use_mmap=True,
                use_mlock=False,

                verbose=False
            )

            self.loaded = True

            logger.info("Qwen2.5-3B GGUF model loaded successfully.")
            logger.info("Inference device: CPU")

    def generate(self, messages, max_new_tokens):
        """
        Generate a response using Qwen2.5-3B.
        """

        self.load()

        with self.generate_lock:

            response = self.model.create_chat_completion(
                messages=messages,

                max_tokens=max_new_tokens,

                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.08,

                stop=[
                    "<|im_end|>",
                    "<|endoftext|>"
                ]
            )

        content = response["choices"][0]["message"]["content"]

        if not content:
            raise RuntimeError("The model returned an empty response.")

        return content.strip()


# Create one shared model service for the application.
service = ModelService()


# ============================================================
# 7. INPUT VALIDATION
# ============================================================

SUPPORTED_QUESTION_TYPES = {
    "multiple_choice",
    "checkbox",
    "rating",
    "scale",
    "yes_no",
    "short_answer",
    "paragraph",
    "dropdown"
}


def get_request_value(data, *keys, default=None):
    """
    Return the first non-empty value found among possible keys.
    """

    for key in keys:
        value = data.get(key)

        if value is not None and value != "":
            return value

    return default


def validate_survey_input(data):
    """
    Validate and normalize the incoming survey requirements.
    """

    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    topic = get_request_value(
        data,
        "topic",
        "survey_topic",
        "title"
    )

    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("Survey topic is required.")

    topic = topic.strip()

    if len(topic) > 500:
        raise ValueError("Survey topic must not exceed 500 characters.")

    purpose = get_request_value(
        data,
        "purpose",
        "description",
        default="General research and information gathering"
    )

    audience = get_request_value(
        data,
        "target_audience",
        "audience",
        default="General audience"
    )

    language = get_request_value(
        data,
        "language",
        default="English"
    )

    additional_instructions = get_request_value(
        data,
        "additional_instructions",
        "instructions",
        default=""
    )

    question_count = get_request_value(
        data,
        "question_count",
        "num_questions",
        "number_of_questions",
        default=10
    )

    try:
        question_count = int(question_count)
    except (ValueError, TypeError):
        raise ValueError("Question count must be a number.")

    if question_count < 1 or question_count > 30:
        raise ValueError(
            "Question count must be between 1 and 30."
        )

    if not isinstance(audience, str):
        raise ValueError("Target audience must be text.")

    if not isinstance(purpose, str):
        raise ValueError("Survey purpose must be text.")

    if not isinstance(language, str):
        raise ValueError("Language must be text.")

    if not isinstance(additional_instructions, str):
        raise ValueError("Additional instructions must be text.")

    if len(audience) > 300:
        raise ValueError("Target audience is too long.")

    if len(purpose) > 1000:
        raise ValueError("Survey purpose is too long.")

    if len(additional_instructions) > 1500:
        raise ValueError("Additional instructions are too long.")

    question_types = get_request_value(
        data,
        "question_types",
        default=["multiple_choice", "short_answer"]
    )

    if isinstance(question_types, str):
        question_types = [question_types]

    if not isinstance(question_types, list) or not question_types:
        raise ValueError("At least one question type is required.")

    normalized_types = []

    for question_type in question_types:

        if not isinstance(question_type, str):
            raise ValueError("Question types must be text.")

        question_type = question_type.strip().lower()

        if question_type not in SUPPORTED_QUESTION_TYPES:
            raise ValueError(
                f"Unsupported question type: {question_type}"
            )

        if question_type not in normalized_types:
            normalized_types.append(question_type)

    include_demographics = data.get(
        "include_demographics",
        False
    )

    if not isinstance(include_demographics, bool):
        raise ValueError("include_demographics must be true or false.")

    return {
        "topic": topic,
        "purpose": purpose.strip(),
        "target_audience": audience.strip(),
        "language": language.strip(),
        "question_count": question_count,
        "question_types": normalized_types,
        "additional_instructions": additional_instructions.strip(),
        "include_demographics": include_demographics
    }


# ============================================================
# 8. PROMPT BUILDER
# ============================================================

def make_prompt(survey_data):
    """
    Convert validated survey requirements into a model prompt.
    """

    question_types = ", ".join(
        survey_data["question_types"]
    )

    prompt = f"""
Create a complete survey using these requirements:

SURVEY TOPIC:
{survey_data["topic"]}

PURPOSE:
{survey_data["purpose"]}

TARGET AUDIENCE:
{survey_data["target_audience"]}

LANGUAGE:
{survey_data["language"]}

NUMBER OF QUESTIONS:
{survey_data["question_count"]}

ALLOWED QUESTION TYPES:
{question_types}

INCLUDE DEMOGRAPHIC QUESTIONS:
{"Yes" if survey_data["include_demographics"] else "No"}

ADDITIONAL INSTRUCTIONS:
{survey_data["additional_instructions"] or "None"}

REQUIREMENTS:

- Generate exactly {survey_data["question_count"]} questions.
- Use the allowed question types.
- Make questions relevant to the topic and purpose.
- Avoid duplicate and leading questions.
- Make all questions clear and easy to understand.
- Use the requested language.
- Include answer choices where appropriate.
- Set required to true or false for every question.
- Give each question a unique sequential ID starting at 1.
- Return the result using the exact JSON structure in the system instructions.
- Return valid JSON only.
"""

    return prompt.strip()


# ============================================================
# 9. JSON RESPONSE PARSER
# ============================================================

def parse_model_json(raw_response):
    """
    Extract a JSON object from the model response.
    Handles accidental Markdown code fences.
    """

    if not raw_response:
        raise ValueError("Empty model response.")

    cleaned = raw_response.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Attempt to extract the first complete JSON object.
    start = cleaned.find("{")

    if start == -1:
        raise ValueError("No JSON object found in model response.")

    decoder = json.JSONDecoder()

    try:
        parsed, _ = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON generated by model: {exc}"
        )

    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")

    return parsed


# ============================================================
# 10. NORMALIZE AND VALIDATE GENERATED SURVEY
# ============================================================

def normalize_generated_survey(survey, survey_data):
    """
    Ensure the generated survey follows the expected structure.
    """

    if not isinstance(survey, dict):
        raise ValueError("Generated survey is not a JSON object.")

    title = survey.get("title", survey_data["topic"])
    description = survey.get("description", "")
    questions = survey.get("questions", [])

    if not isinstance(title, str):
        title = survey_data["topic"]

    if not isinstance(description, str):
        description = ""

    if not isinstance(questions, list):
        raise ValueError("Generated questions must be a list.")

    normalized_questions = []

    for index, question in enumerate(questions, start=1):

        if not isinstance(question, dict):
            continue

        question_text = question.get("question", "")

        if not isinstance(question_text, str):
            continue

        question_text = question_text.strip()

        if not question_text:
            continue

        question_type = question.get(
            "type",
            "short_answer"
        )

        if question_type not in SUPPORTED_QUESTION_TYPES:
            question_type = "short_answer"

        options = question.get("options", [])

        if not isinstance(options, list):
            options = []

        options = [
            str(option).strip()
            for option in options
            if option is not None and str(option).strip()
        ]

        required = question.get("required", True)

        if not isinstance(required, bool):
            required = True

        normalized_questions.append({
            "id": index,
            "question": question_text,
            "type": question_type,
            "required": required,
            "options": options
        })

    if not normalized_questions:
        raise ValueError("The model did not generate valid questions.")

    # Do not silently return an incomplete survey.
    if len(normalized_questions) != survey_data["question_count"]:
        raise ValueError(
            f"Expected {survey_data['question_count']} questions, "
            f"but the model generated {len(normalized_questions)}. "
            "Try generating again or requesting fewer questions."
        )

    return {
        "title": title.strip() or survey_data["topic"],
        "description": description.strip(),
        "topic": survey_data["topic"],
        "target_audience": survey_data["target_audience"],
        "questions": normalized_questions
    }


# ============================================================
# 11. HEALTH CHECK ENDPOINT
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "running",
        "application": "AI Survey Generator API",
        "base_model": "Qwen2.5-3B-Instruct",
        "adapter": "Merged trained LoRA",
        "quantization": "Q4_K_M",
        "device": service.device,
        "model_loaded": service.loaded,
        "model_exists": MODEL_GGUF_PATH.is_file()
    }), 200


# ============================================================
# 12. API INFORMATION ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "message": "AI Survey Generator API is running.",
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate"
        },
        "method": "POST",
        "authentication": (
            "X-API-Key required"
            if API_KEY
            else "Disabled"
        )
    }), 200


# ============================================================
# 13. GENERATE SURVEY ENDPOINT
# ============================================================

@app.route("/api/generate", methods=["POST"])
def generate_survey():

    try:

        # ----------------------------------------
        # Read request
        # ----------------------------------------

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "A valid JSON request body is required."
            }), 400

        # ----------------------------------------
        # Validate requirements
        # ----------------------------------------

        survey_data = validate_survey_input(data)

        # ----------------------------------------
        # Build model prompt
        # ----------------------------------------

        user_prompt = make_prompt(survey_data)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # ----------------------------------------
        # Determine generation length
        # ----------------------------------------

        requested_tokens = data.get(
            "max_new_tokens",
            DEFAULT_MAX_NEW_TOKENS
        )

        try:
            requested_tokens = int(requested_tokens)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "max_new_tokens must be a number."
            }), 400

        max_new_tokens = max(
            100,
            min(requested_tokens, MAX_ALLOWED_NEW_TOKENS)
        )

        # ----------------------------------------
        # Generate response
        # ----------------------------------------

        logger.info(
            "Generating survey: topic=%s, questions=%s",
            survey_data["topic"],
            survey_data["question_count"]
        )

        raw_response = service.generate(
            messages,
            max_new_tokens
        )

        # ----------------------------------------
        # Parse JSON
        # ----------------------------------------

        parsed_survey = parse_model_json(raw_response)

        # ----------------------------------------
        # Validate generated survey
        # ----------------------------------------

        final_survey = normalize_generated_survey(
            parsed_survey,
            survey_data
        )

        logger.info("Survey generated successfully.")

        return jsonify({
            "success": True,
            "message": "Survey generated successfully.",
            "survey": final_survey
        }), 200

    except ValueError as exc:

        logger.warning("Survey validation error: %s", exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400

    except FileNotFoundError as exc:

        logger.error("Model file error: %s", exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500

    except Exception as exc:

        logger.exception("Survey generation failed.")

        return jsonify({
            "success": False,
            "error": "Survey generation failed.",
            "details": str(exc)
        }), 500


# ============================================================
# 14. ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "success": False,
        "error": "Endpoint not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({
        "success": False,
        "error": "HTTP method not allowed."
    }), 405


@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "success": False,
        "error": "Request body is too large."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({
        "success": False,
        "error": "Internal server error."
    }), 500


# ============================================================
# 15. START APPLICATION
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("AI SURVEY GENERATOR")
    logger.info("=" * 60)

    logger.info("Model: Qwen2.5-3B-Instruct")
    logger.info("Adapter: Merged trained LoRA")
    logger.info("Quantization: Q4_K_M")
    logger.info("Device: CPU")
    logger.info("Context length: %s", N_CTX)
    logger.info("CPU threads: %s", N_THREADS)
    logger.info("Model file: %s", MODEL_GGUF_PATH)
    logger.info("Model file exists: %s", MODEL_GGUF_PATH.is_file())

    logger.info("Starting Flask server on %s:%s", HOST, PORT)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )
