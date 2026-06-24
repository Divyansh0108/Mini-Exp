from pathlib import Path

GENERATION_MODELS = [
    "llama3.1:8b",
    "mistral:7b",
    "deepseek-r1:7b",
]

JUDGE_MODEL = "qwen3:14b"

NUM_SAMPLES = 50
RANDOM_SEED = 42

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = ROOT_DIR / "plots"
STYLE_DIR = ROOT_DIR / "style"
VISUALS_DIR = ROOT_DIR / "visuals"

ARC_SAMPLE_PATH = DATA_DIR / "arc_sample.json"
STYLE_PATH = STYLE_DIR / "cot_style.mplstyle"

OLLAMA_TEMPERATURE = 0.0
OLLAMA_MAX_TOKENS = 512
OLLAMA_JUDGE_TOKENS = 10
OLLAMA_TIMEOUT_SECONDS = 120

VALID_ANSWERS = ["A", "B", "C", "D"]
VALID_JUDGE_ANSWERS = VALID_ANSWERS + ["X"]

FAITHFUL_CORRECT = "faithful_correct"
FAITHFUL_INCORRECT = "faithful_incorrect"
UNFAITHFUL = "unfaithful"
AMBIGUOUS = "ambiguous"

MODEL_DISPLAY_NAMES = {
    "llama3.1:8b": "Llama 3.1 8B",
    "mistral:7b": "Mistral 7B",
    "deepseek-r1:7b": "DeepSeek-R1 7B",
}


def model_slug(model_name: str) -> str:
    return model_name.replace(":", "_").replace(".", "_")
