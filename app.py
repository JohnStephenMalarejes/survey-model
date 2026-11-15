import os
import re
import copy
import hmac
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

def resolve_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


MODEL_GGUF_PATH = resolve_path(
    os.getenv("MODEL_GGUF_PATH", "./model/ai_survey_qwen_3b_q4.gguf")
)

# Optional: a separate model for research insights (for example the
# plain Qwen2.5-3B-Instruct Q4 GGUF). If not set, the survey model is
# reused and loaded only once.
INSIGHT_MODEL_GGUF_PATH = resolve_path(
    os.getenv("INSIGHT_MODEL_GGUF_PATH", str(MODEL_GGUF_PATH))
)

# Insights need more context than survey generation because the
# statistics (including follow-up answers) are part of the prompt. If the
# detailed overall insight returns truncated JSON, raise this further in
# your .env. NOTE: a value set in .env overrides this default.
N_CTX = int(os.getenv("N_CTX", "6144"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_BATCH = int(os.getenv("N_BATCH", "64"))

# Safety room kept free in the context window, and the smallest reply
# (in tokens) worth generating for a research insight.
CONTEXT_MARGIN = 64
MIN_REPLY_TOKENS = 300

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "700"))
MAX_ALLOWED_NEW_TOKENS = int(os.getenv("MAX_ALLOWED_NEW_TOKENS", "900"))

# Single-question insights stay short.
INSIGHT_MAX_NEW_TOKENS = int(os.getenv("INSIGHT_MAX_NEW_TOKENS", "700"))
# The overall survey insight is longer and more detailed.
INSIGHT_SURVEY_MAX_NEW_TOKENS = int(os.getenv("INSIGHT_SURVEY_MAX_NEW_TOKENS", "1500"))
# Follow-up answers share this budget with the main question statistics.
INSIGHT_MAX_STATS_CHARS = int(os.getenv("INSIGHT_MAX_STATS_CHARS", "7000"))

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
        "http://localhost:3000,http://localhost:5173"
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

    if not hmac.compare_digest(supplied_key, API_KEY):
        return jsonify({
            "success": False,
            "error": "Unauthorized. Invalid or missing API key."
        }), 401

    return None


# ============================================================
# 5. SYSTEM PROMPTS
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


INSIGHT_RULES = """
RULES:

1. Use only the numbers and answer labels provided. Never invent
   figures, answers, respondents, or quotes.
2. Responses describe what participants selected. They do not prove
   why. Present explanations as hypotheses, using words such as
   "may", "could", or "one possible explanation".
3. Do not claim statistical significance, representativeness, or
   causation.
4. Mention low response counts or low answer rates when relevant.
5. Each question has an answerKind. For "choice" and "open_text"
   questions, topAnswers lists the actual answers with counts. Only
   name an answer or theme that appears in topAnswers, and quote
   answers as written. For open_text questions, say so when most
   answers are non-answers such as "I don't know" or "none", and do
   not invent themes. If topAnswers is empty, you have no information
   about what respondents said: report only how many answered.
6. Every sentence must be backed by the statistics. Do not pad with
   generic advice that ignores the data.
7. Return valid JSON only, with no Markdown and no text outside
   the JSON.
8. The statistics may include a "followUps" list. Each item is an
   extra question that was asked ONLY to respondents whose answer to
   an earlier question triggered it. parentQuestion is the question it
   follows, triggeredByAnswer shows which answers triggered it (for
   example {"No": 4}), and answersCount counts only those
   respondents, never the whole sample. topAnswers lists their written
   answers with counts. Use follow-ups to explain or qualify the
   parentQuestion, quote only answers that appear in topAnswers, and
   never present follow-up counts as results for all respondents. If
   there is no "followUps" list, do not mention follow-ups.
"""


INSIGHT_QUESTION_PROMPT = """
You are a careful survey research analyst.

You receive pre-computed statistics for ONE survey question. Write a
short research insight based ONLY on those statistics.
""" + INSIGHT_RULES + """
9. Keep each field to 2-5 sentences, in plain language.

REQUIRED JSON STRUCTURE:

{
    "summary": "Executive summary of the main observed patterns.",
    "discussion": "Possible explanations, stated as hypotheses.",
    "recommendation": "Concrete next research steps.",
    "limitations": "Limits of this data and analysis."
}
"""


INSIGHT_SURVEY_PROMPT = """
You are a careful survey research analyst writing the overall report
for a whole survey.

You receive pre-computed statistics for every question. Write a
detailed, well-organized research insight based ONLY on those
statistics.
""" + INSIGHT_RULES + """
9. Be thorough. Cover every question that has answers, not just the
   most striking ones. Compare questions where the data allows it,
   and point out contradictions (for example, a question answered
   positively but follow-up answers that suggest uncertainty).
10. If followUps are provided, cover every follow-up that has answers.
   Tie each one back to its parentQuestion and say which answer
   triggered it. Write at least one finding about follow-up answers,
   and use them in the discussion to explain the main results.
11. Length per field: summary 5-8 sentences; findings 5-12 items;
   discussion 6-9 sentences; recommendation 5-8 sentences;
   limitations 4-6 sentences.

REQUIRED JSON STRUCTURE:

{
    "summary": "Overview: response count, answer completeness, and the main patterns across the survey.",
    "findings": [
        "One finding per question or theme, citing counts and percentages.",
        "Another finding."
    ],
    "discussion": "Possible explanations and connections between questions, stated as hypotheses.",
    "recommendation": "Concrete, prioritized next research steps tied to the findings.",
    "limitations": "Limits of this data and analysis, including sample size."
}
"""


# ============================================================
# 6. MODEL SERVICE
# ============================================================

class ModelService:

    def __init__(self, model_path, label):
        self.model_path = model_path
        self.label = label

        self.model = None
        self.loaded = False

        self.load_lock = threading.Lock()
        self.generate_lock = threading.Lock()

        self.device = "cpu"

    def load(self):
        """
        Load a quantized GGUF model.
        """

        if self.loaded:
            return

        with self.load_lock:

            if self.loaded:
                return

            logger.info("Loading %s model...", self.label)
            logger.info("Model path: %s", self.model_path)

            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"GGUF model not found: {self.model_path}. "
                    "Place your quantized model in the model folder "
                    "or update the path in your .env file."
                )

            self.model = Llama(
                model_path=str(self.model_path),

                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_batch=N_BATCH,

                n_gpu_layers=0,

                use_mmap=True,
                use_mlock=False,

                verbose=False
            )

            self.loaded = True

            logger.info("%s model loaded successfully.", self.label)

    def count_tokens(self, messages):
        """
        Approximate number of prompt tokens for a chat (Qwen/ChatML
        layout). Used to make sure prompt + reply fit in N_CTX.
        """

        self.load()

        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            for m in messages
        ) + "<|im_start|>assistant\n"

        return len(
            self.model.tokenize(
                text.encode("utf-8"),
                add_bos=False,
                special=True
            )
        )

    def generate(self, messages, max_new_tokens, temperature=0.3, seed=None):
        """
        Generate a chat response.
        """

        self.load()

        with self.generate_lock:

            response = self.model.create_chat_completion(
                messages=messages,

                max_tokens=max_new_tokens,

                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.08,
                seed=seed,
                stop=[
                    "<|im_end|>",
                    "<|endoftext|>"
                ]
            )

        content = response["choices"][0]["message"]["content"]

        if not content:
            raise RuntimeError("The model returned an empty response.")

        return content.strip()


# Survey generation model.
service = ModelService(MODEL_GGUF_PATH, "AI Survey Generator")

# Insight model. Reuses the same loaded model when the paths match,
# so the memory cost is not doubled.
if INSIGHT_MODEL_GGUF_PATH == MODEL_GGUF_PATH:
    insight_service = service
else:
    insight_service = ModelService(INSIGHT_MODEL_GGUF_PATH, "Research Insight")


# ============================================================
# 7. INPUT VALIDATION (SURVEY GENERATION)
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
# 8. PROMPT BUILDERS
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
# 11. RESEARCH INSIGHT HELPERS
# ============================================================

INSIGHT_FIELDS = ("summary", "discussion", "recommendation", "limitations")


def as_text(value):
    """
    Convert a model field to plain text. Small models sometimes
    return a list of sentences instead of a string.
    """

    if value is None:
        return ""

    if isinstance(value, (list, tuple)):
        return " ".join(
            str(item).strip() for item in value if str(item).strip()
        )

    return str(value).strip()


def compact_stats(stats, limit):
    """
    Serialize statistics as compact JSON that fits the prompt budget.

    First shortens each topAnswers list (questions and follow-ups),
    then drops trailing follow-ups, then trailing questions, if it is
    still too long. Returns the JSON text, the number of questions
    omitted and the number of follow-ups omitted.
    """

    data = copy.deepcopy(stats)

    def dump():
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    questions = data.get("questions")

    # Keep a reference to the list stored in data so popping from it
    # shrinks the serialized output. Anything that is not a list is
    # ignored and removed.
    follow_ups = data.get("followUps")

    if not isinstance(follow_ups, list):
        follow_ups = []
        data.pop("followUps", None)

    if not isinstance(questions, list):
        text = dump()
        if len(text) > limit:
            raise ValueError("Statistics are too large to analyze.")
        return text, 0, 0

    for cap in (6, 4, 3, 2, 0):
        for item in [*questions, *follow_ups]:
            if isinstance(item, dict) and isinstance(
                item.get("topAnswers"), list
            ):
                item["topAnswers"] = item["topAnswers"][:cap]

        if len(dump()) <= limit:
            return dump(), 0, 0

    # Follow-ups are supplementary to the main questions, so they are
    # dropped first.
    omitted_follow_ups = 0

    while follow_ups and len(dump()) > limit:
        follow_ups.pop()
        omitted_follow_ups += 1

    omitted = 0

    while questions and len(dump()) > limit:
        questions.pop()
        omitted += 1

    if not questions:
        raise ValueError("Statistics are too large to analyze.")

    if omitted:
        data["omittedQuestions"] = omitted

    if omitted_follow_ups:
        data["omittedFollowUps"] = omitted_follow_ups

    return dump(), omitted, omitted_follow_ups


def validate_insight_input(data):
    """
    Validate the statistics payload sent by the frontend.
    """

    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")

    scope = data.get("scope", "survey")

    if scope not in ("survey", "question"):
        raise ValueError("scope must be 'survey' or 'question'.")

    stats = data.get("stats")

    if not isinstance(stats, dict) or not stats:
        raise ValueError("stats is required and must be a JSON object.")

    response_count = stats.get("validResponseCount", stats.get("responseCount"))

    if isinstance(response_count, (int, float)) and response_count <= 0:
        raise ValueError("There are no responses to analyze.")

    return scope, stats


# ============================================================
# 12. HEALTH CHECK ENDPOINT
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
        "context_length": N_CTX,
        "model_loaded": service.loaded,
        "model_exists": MODEL_GGUF_PATH.is_file(),
        "insight_model_exists": INSIGHT_MODEL_GGUF_PATH.is_file(),
        "insight_model_shared": insight_service is service
    }), 200


# ============================================================
# 13. API INFORMATION ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "message": "AI Survey Generator API is running.",
        "endpoints": {
            "health": "/health",
            "generate": "/api/generate",
            "insight": "/api/insight",
            "followups": "/api/followups"
        },
        "method": "POST",
        "authentication": (
            "X-API-Key required"
            if API_KEY
            else "Disabled"
        )
    }), 200


# ============================================================
# 14. GENERATE SURVEY ENDPOINT
# ============================================================

@app.route("/api/generate", methods=["POST"])
def generate_survey():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "A valid JSON request body is required."
            }), 400

        survey_data = validate_survey_input(data)

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

        logger.info(
            "Generating survey: topic=%s, questions=%s",
            survey_data["topic"],
            survey_data["question_count"]
        )

        raw_response = service.generate(
            messages,
            max_new_tokens
        )

        parsed_survey = parse_model_json(raw_response)

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
# 15. RESEARCH INSIGHT ENDPOINT
# ============================================================

@app.route("/api/insight", methods=["POST"])
def generate_insight():
    """
    Turn pre-computed survey statistics into a written research
    insight. The model only narrates the numbers it is given.

    scope="survey"   -> long, detailed overall report (with findings)
    scope="question" -> short insight for one question
    """

    try:

        data = request.get_json(silent=True)

        scope, stats = validate_insight_input(data)

        system_prompt = (
            INSIGHT_SURVEY_PROMPT if scope == "survey" else INSIGHT_QUESTION_PROMPT
        )
        requested_tokens = (
            INSIGHT_SURVEY_MAX_NEW_TOKENS
            if scope == "survey"
            else INSIGHT_MAX_NEW_TOKENS
        )

        # Shrink the statistics until prompt + reply fit in the model's
        # context window (N_CTX), instead of failing with "Requested
        # tokens exceed context window".
        stats_limit = INSIGHT_MAX_STATS_CHARS
        min_stats_limit = 1200

        while True:
            stats_json, omitted, omitted_follow_ups = compact_stats(
                stats, stats_limit
            )

            user_prompt = (
                f"Scope: {scope}\n"
                "Statistics (JSON):\n"
                f"{stats_json}\n\n"
                "Write the research insight now. Return JSON only."
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]

            prompt_tokens = insight_service.count_tokens(messages)
            room = N_CTX - prompt_tokens - CONTEXT_MARGIN

            if room >= requested_tokens or stats_limit <= min_stats_limit:
                break

            stats_limit = max(min_stats_limit, int(stats_limit * 0.75))

        if room < MIN_REPLY_TOKENS:
            raise ValueError(
                f"The statistics are too large for the model's context "
                f"window ({N_CTX} tokens). Set N_CTX=6144 in your .env "
                "file and restart the AI service."
            )

        max_tokens = min(requested_tokens, room)
        reply_clamped = max_tokens < requested_tokens

        if reply_clamped:
            logger.warning(
                "Reply limited to %s tokens (wanted %s) because N_CTX=%s "
                "is small. Raise N_CTX in .env for fuller insights.",
                max_tokens,
                requested_tokens,
                N_CTX
            )

        logger.info(
            "Generating insight: scope=%s, stats_chars=%s, omitted=%s, "
            "omitted_followups=%s, max_tokens=%s",
            scope,
            len(stats_json),
            omitted,
            omitted_follow_ups,
            max_tokens
        )

        raw_response = insight_service.generate(
            messages,
            max_tokens,
            temperature=0.2
        )

        try:
            parsed = parse_model_json(raw_response)
        except ValueError:
            if reply_clamped:
                raise ValueError(
                    "The AI reply was cut off because the model's context "
                    f"window is too small (N_CTX={N_CTX}). Set N_CTX=6144 "
                    "in your .env file and restart the AI service."
                )
            raise

        insight = {
            field: as_text(parsed.get(field))
            for field in INSIGHT_FIELDS
        }

        findings = parsed.get("findings")

        if isinstance(findings, str):
            findings = [findings]

        insight["findings"] = (
            [str(f).strip() for f in findings if str(f).strip()]
            if isinstance(findings, list)
            else []
        )

        if not insight["summary"]:
            raise ValueError("The model did not return a summary.")

        if not insight["limitations"]:
            insight["limitations"] = (
                "This insight describes submitted responses only and "
                "does not establish causes or representativeness."
            )

        logger.info("Insight generated successfully.")

        return jsonify({
            "success": True,
            "insight": insight,
            "model": "Qwen2.5-3B-Instruct (local, Q4_K_M)",
            "omittedQuestions": omitted,
            "omittedFollowUps": omitted_follow_ups
        }), 200

    except ValueError as exc:

        logger.warning("Insight error: %s", exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 422

    except FileNotFoundError as exc:

        logger.error("Model file error: %s", exc)

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500

    except Exception as exc:

        logger.exception("Insight generation failed.")

        return jsonify({
            "success": False,
            "error": "Insight generation failed.",
            "details": str(exc)
        }), 500


# ============================================================
# 15b. CONDITIONAL FOLLOW-UP ENDPOINT
# ============================================================

FOLLOWUP_SYSTEM_PROMPT = """
You write follow-up questions for a survey.

A respondent answered an earlier question with "Yes" or "No". Write
follow-up questions that fit THAT answer only.

RULES:

1. Each follow-up must relate to the ORIGINAL QUESTION and to the
   specific answer given. Base it only on the wording of that
   question, not on any wider topic.
2. If the answer is "Yes": the respondent agrees or has experienced
   it. Ask about their experience: details, how well it worked, what
   could be improved.
3. If the answer is "No": the respondent says it is NOT true for
   them, or they disagree. NEVER assume the thing exists, or that
   they used, received, or experienced it. Ask WHY not, what is
   missing or in the way, what they would need for the answer to
   change, or what they would want instead.
4. Make them open-ended, neutral, and not leading.
5. Keep each under 25 words, in simple language.
6. Do not repeat the original question or each other.
7. Return valid JSON only, with no Markdown:

{"followups": ["Follow-up question 1?", "Follow-up question 2?"]}
"""

# Worked examples (a different topic on purpose, so the model copies
# the pattern and not the content). One "No" and one "Yes".
FOLLOWUP_EXAMPLES = [
    {"role": "user", "content": (
        "Original question: Does your neighborhood have a public library?\n"
        "Respondent answered: No\n\n"
        "Meaning: the respondent said this is NOT true for them. "
        "Do not assume it exists or that they used it.\n\n"
        "Write exactly 2 NEW follow-up question(s). Return JSON only.")},
    {"role": "assistant", "content": json.dumps({"followups": [
        "What makes it hard for you to access a library?",
        "What would you want a library or similar service to offer you?"
    ]})},
    {"role": "user", "content": (
        "Original question: Does your neighborhood have a public library?\n"
        "Respondent answered: Yes\n\n"
        "Meaning: the respondent said this is TRUE for them. "
        "Ask about their experience with it.\n\n"
        "Write exactly 2 NEW follow-up question(s). Return JSON only.")},
    {"role": "assistant", "content": json.dumps({"followups": [
        "How often do you use the library?",
        "What could the library do to serve you better?"
    ]})},
]

ANSWER_MEANING = {
    "yes": "the respondent said this is TRUE for them. Ask about their experience with it.",
    "no": "the respondent said this is NOT true for them. Do not assume it exists or that they used it.",
}


def extract_followups(parsed, seen, count):
    items = parsed.get("followups")
    if items is None and isinstance(parsed.get("questions"), list):
        items = [q.get("question") if isinstance(q, dict) else q
                 for q in parsed["questions"]]
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", item).strip()
        if not text or len(text) > 300 or text.lower() in seen:
            continue
        seen.add(text.lower())
        out.append(text)
        if len(out) == count:
            break
    return out


@app.route("/api/followups", methods=["POST"])
def generate_followups():
    """
    Generate follow-up questions for one Yes/No branch of a
    conditional question.
    """

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError("A valid JSON request body is required.")

        question = data.get("question")
        answer = str(data.get("answer", "")).strip().lower()

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is required.")

        question = question.strip()

        if len(question) > 500:
            raise ValueError("question must not exceed 500 characters.")

        if answer not in ("yes", "no"):
            raise ValueError("answer must be 'Yes' or 'No'.")

        try:
            count = int(data.get("count", 3))
        except (ValueError, TypeError):
            raise ValueError("count must be a number.")

        count = max(1, min(count, 5))

        # survey_title / survey_description may still be sent by older
        # clients. They are deliberately NOT used: follow-ups are written
        # from the question text alone so a broad survey topic cannot
        # pull the model away from the question being asked.

        # Previously shown follow-ups to avoid, plus an optional seed.
        avoid = data.get("avoid", [])
        avoid = (
            [a.strip()[:300] for a in avoid if isinstance(a, str) and a.strip()][:6]
            if isinstance(avoid, list) else []
        )
        seed = data.get("seed")
        seed = seed if isinstance(seed, int) and 0 <= seed < 2**31 else None

        avoid_block = ""
        if avoid:
            avoid_block = (
                "Do NOT reuse or rephrase any of these existing follow-ups:\n"
                + "\n".join(f"- {a}" for a in avoid) + "\n\n"
            )

        # Same layout as the worked examples in FOLLOWUP_EXAMPLES, so the
        # model sees a consistent format.
        user_prompt = (
            f"Original question: {question}\n"
            f"Respondent answered: {answer.capitalize()}\n\n"
            f"Meaning: {ANSWER_MEANING[answer]}\n\n"
            f"{avoid_block}"
            f"Write exactly {count} NEW follow-up question(s). Return JSON only."
        )

        messages = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            *FOLLOWUP_EXAMPLES,
            {"role": "user", "content": user_prompt}
        ]

        logger.info("Generating follow-ups: answer=%s, count=%s", answer, count)

        seen = {question.lower()} | {a.lower() for a in avoid}
        followups = []

        for attempt in range(2):
            raw_response = service.generate(
                messages, 450,
                temperature=0.6 + 0.1 * attempt,
                seed=None if seed is None else seed + attempt
            )
            try:
                parsed = parse_model_json(raw_response)
            except ValueError:
                continue
            followups = extract_followups(parsed, seen, count)
            if followups:
                break

        if not followups:
            raise ValueError("The model only returned questions you already have. Try again.")

        return jsonify({"success": True, "followups": followups}), 200

    except ValueError as exc:
        logger.warning("Follow-up error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 422

    except FileNotFoundError as exc:
        logger.error("Model file error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 500

    except Exception as exc:
        logger.exception("Follow-up generation failed.")
        return jsonify({
            "success": False,
            "error": "Follow-up generation failed.",
            "details": str(exc)
        }), 500


# ============================================================
# 16. ERROR HANDLERS
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
# 17. START APPLICATION
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("AI SURVEY GENERATOR + RESEARCH INSIGHTS")
    logger.info("=" * 60)

    logger.info("Model: Qwen2.5-3B-Instruct")
    logger.info("Quantization: Q4_K_M")
    logger.info("Device: CPU")
    logger.info("Context length: %s", N_CTX)
    logger.info("CPU threads: %s", N_THREADS)
    logger.info("Survey model: %s", MODEL_GGUF_PATH)
    logger.info("Insight model: %s", INSIGHT_MODEL_GGUF_PATH)
    logger.info("Survey model exists: %s", MODEL_GGUF_PATH.is_file())
    logger.info("Insight model exists: %s", INSIGHT_MODEL_GGUF_PATH.is_file())

    logger.info("Starting Flask server on %s:%s", HOST, PORT)

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        threaded=True,
        use_reloader=False
    )