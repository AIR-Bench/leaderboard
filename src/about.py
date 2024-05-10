from dataclasses import dataclass
from enum import Enum


@dataclass
class Task:
    name: str  # qa, long_doc

@dataclass
class Metric:
    name: str  # ndcg_at_1

@dataclass
class Language:
    name: str  # en, zh


@dataclass
class Domain:
    name: str  # law, wiki


@dataclass
class EmbeddingModel:
    full_name: str  # jinaai/jina-embeddings-v2-en-base
    org: str  # jinaai
    model: str  # jina-embeddings-v2-en-base
    size: int  # size (millions of parameters)
    dim: int  # output dimensions
    max_tokens: int  # max tokens
    model_type: str  # open, proprietary, sentence transformers



NUM_FEWSHOT = 0 # Change with your few shot
# ---------------------------------------------------



# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">AIR-Bench</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
"""

# Which evaluations are you running? how can people reproduce what you have?
LLM_BENCHMARKS_TEXT = f"""
## How it works

## Reproducibility
To reproduce our results, here is the commands you can run:

"""

EVALUATION_QUEUE_TEXT = """
## Some good practices before submitting a model

### 1) Make sure you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
If this step fails, follow the error messages to debug your model before submitting it. It's likely your model has been improperly uploaded.

Note: make sure your model is public!
Note: if your model needs `use_remote_code=True`, we do not support this option yet but we are working on adding it, stay posted!

### 2) Convert your model weights to [safetensors](https://huggingface.co/docs/safetensors/index)
It's a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!

### 3) Make sure your model has an open license!
This is a leaderboard for Open LLMs, and we'd love for as many people as possible to know they can use your model 🤗

### 4) Fill up your model card
When we add extra information about models to the leaderboard, it will be automatically taken from the model card

## In case of model failure
If your model is displayed in the `FAILED` category, its execution stopped.
Make sure you have followed the above steps first.
If everything is done, check you can launch the EleutherAIHarness on your model locally, using the above command without modifications (you can add `--limit` to limit the number of examples per task).
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
"""
