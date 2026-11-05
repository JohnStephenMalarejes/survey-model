# AI Survey Generator Flask API --- Local Setup (Windows)

> **Important:** Follow these steps only for your personal local copy of
> the AI Survey Generator Flask API project. Do not modify the original
> or shared project. All installations, configurations, and generated
> files must apply only to your local project directory.

This guide assumes Windows, Python 3.11.9 (64-bit), VS Code, and a
CPU-based GGUF model using `llama-cpp-python`.

## Step 1. Open your project folder

Open your local `ai_survey_flask_api` folder in VS Code. Select
**Terminal → New Terminal** and make sure the terminal is inside your
project directory.


## Step 2. Create a virtual environment

Run:
python -m venv .venv

This creates a `.venv` folder inside your project.

## Step 3. Activate the virtual environment

.venv\Scripts\activate.bat

You should see `(.venv)` at the beginning of your terminal prompt.

## Step 4. Upgrade pip

With `.venv` activated, run:

python -m pip install --upgrade pip


## Step 6. Install the requirements

Run:

python -m pip install --no-cache-dir --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt

This requests a precompiled CPU wheel for `llama-cpp-python`, avoiding a
local C++ build when a compatible wheel is available. If no compatible
wheel exists for your Python version and Windows architecture,
installation may still require Microsoft C++ Build Tools or a compatible
Python/package combination.

## Step 7. Verify the installed packages

python -c "import flask, flask_cors, dotenv, llama_cpp; print('All dependencies installed successfully!')"

Expected output:

All dependencies installed successfully!


## Step 8. Check your model file

Make sure the actual GGUF model file exists here:

model/ai_survey_qwen_3b_q4.gguf

Your project should look similar to:

ai_survey_flask_api/
├── app.py
├── requirements.txt
├── .env
├── model/
│   └── ai_survey_qwen_3b_q4.gguf
└── .venv/


The LoRA adapter folder alone is not a substitute for the GGUF model
file.

## Step 9. Configure your `.env` file

Create `.env` in the project root, beside `app.py`, and add:

MODEL_GGUF_PATH=./model/ai_survey_qwen_3b_q4.gguf
N_CTX=1024
N_THREADS=4
N_BATCH=64
DEFAULT_MAX_NEW_TOKENS=700
MAX_ALLOWED_NEW_TOKENS=900
API_KEY=
ALLOWED_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=5000


For local testing, `API_KEY=` is empty. If you set an API key, include
the matching `X-API-Key` header in Bruno.

If your computer runs low on memory, try `N_CTX=768` and `N_BATCH=32`,
then restart Flask.

## Step 10. Run the Flask application

Confirm that `.venv` is activated, the GGUF file is in the `model`
folder, and `.env` is configured.

Run:

python app.py


Keep the terminal open. The API should be available at:

http://localhost:5000


