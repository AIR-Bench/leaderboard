import os
from display.formatting import model_hyperlink
from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("TOKEN", "")  # A read/write token for your org

OWNER = "AIR-Bench"  # "nan"  # Change to your org - don't forget to create a results and request dataset, with the correct format!
# ----------------------------------

REPO_ID = f"{OWNER}/leaderboard"
# repo for storing the evaluation results
RESULTS_REPO = f"{OWNER}/eval_results"
# repo for submitting the evaluation
SEARCH_RESULTS_REPO = f"{OWNER}/search_results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval_results")

API = HfApi(token=TOKEN)

BM25_LINK = model_hyperlink("https://github.com/castorini/pyserini", "BM25")

BENCHMARK_VERSION_LIST = [
    "AIR-Bench_24.04",
    # "AIR-Bench_24.05",
]

LATEST_BENCHMARK_VERSION = BENCHMARK_VERSION_LIST[-1]
