import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("TOKEN", "")  # A read/write token for your org

OWNER = (
    "AIR-Bench"  # Change to your org - don't forget to create a results and request dataset, with the correct format!
)
# ----------------------------------

REPO_ID = f"{OWNER}/leaderboard"
# repo for storing the evaluation results
RESULTS_REPO = f"{OWNER}/eval_results"
# repo for submitting the evaluation
SEARCH_RESULTS_REPO = f"{OWNER}/search_results"

# If you set up a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_RESULTS_PATH = os.path.join(CACHE_PATH, "eval_results")

API = HfApi(token=TOKEN)

BENCHMARK_VERSION_LIST = [
    "AIR-Bench_24.04",
    "AIR-Bench_24.05",
]

LATEST_BENCHMARK_VERSION = BENCHMARK_VERSION_LIST[-1]   # Change to the latest benchmark version
DEFAULT_METRIC_QA = "ndcg_at_10"
DEFAULT_METRIC_LONG_DOC = "recall_at_10"
METRIC_LIST = [
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_100",
    "ndcg_at_1000",
    "map_at_1",
    "map_at_3",
    "map_at_5",
    "map_at_10",
    "map_at_100",
    "map_at_1000",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "recall_at_100",
    "recall_at_1000",
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "precision_at_10",
    "precision_at_100",
    "precision_at_1000",
    "mrr_at_1",
    "mrr_at_3",
    "mrr_at_5",
    "mrr_at_10",
    "mrr_at_100",
    "mrr_at_1000",
]
