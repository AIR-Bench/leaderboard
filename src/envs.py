import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("TOKEN")  # A read/write token for your org

OWNER = "nan"  # Change to your org - don't forget to create a results and request dataset, with the correct format!
# ----------------------------------

REPO_ID = f"{OWNER}/leaderboard"
QUEUE_REPO = f"{OWNER}/requests"
RESULTS_REPO = f"{OWNER}/results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_REQUESTS_PATH = "/Users/nanwang/Codes/huggingface/nan/leaderboard/tests/toydata/test_requests"  # os.path.join(CACHE_PATH, "eval-queue")
EVAL_RESULTS_PATH = "/Users/nanwang/Codes/huggingface/nan/leaderboard/tests/toydata/test_results"  #os.path.join(CACHE_PATH, "eval-results")
# EVAL_REQUESTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-queue-bk")
# EVAL_RESULTS_PATH_BACKEND = os.path.join(CACHE_PATH, "eval-results-bk")

API = HfApi(token=TOKEN)
