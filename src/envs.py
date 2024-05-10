import os

from huggingface_hub import HfApi

# Info to change for your repository
# ----------------------------------
TOKEN = os.environ.get("TOKEN")  # A read/write token for your org

OWNER = "nan"  # Change to your org - don't forget to create a results and request dataset, with the correct format!
# ----------------------------------

REPO_ID = f"{OWNER}/leaderboard"
RESULTS_REPO = f"{OWNER}/results"

# If you setup a cache later, just change HF_HOME
CACHE_PATH = os.getenv("HF_HOME", ".")

# Local caches
EVAL_RESULTS_PATH = "/Users/nanwang/Codes/huggingface/nan/results/demo-leaderboard"  #os.path.join(CACHE_PATH, "eval-results")

API = HfApi(token=TOKEN)
