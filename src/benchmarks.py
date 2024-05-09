from dataclasses import dataclass
from enum import Enum


def get_safe_name(name: str):
    """Get RFC 1123 compatible safe name"""
    name = name.replace('-', '_')
    return ''.join(
        character.lower()
        for character in name
        if (character.isalnum() or character == '_'))


dataset_dict = {
    "qa": {
        "wiki": {
            "en": ["wikipedia_20240101", ],
            "zh": ["wikipedia_20240101", ]
        },
        "web": {
            "en": ["mC4", ],
            "zh": ["mC4", ]
        },
        "news": {
            "en": ["CC-News", ],
            "zh": ["CC-News", ]
        },
        "health": {
            "en": ["PubMedQA", ],
            "zh": ["Huatuo-26M", ]
        },
        "law": {
            "en": ["pile-of-law", ],
            "zh": ["flk_npc_gov_cn", ]
        },
        "finance": {
            "en": ["Reuters-Financial", ],
            "zh": ["FinCorpus", ]
        },
        "arxiv": {
            "en": ["Arxiv", ]},
    },
    "long_doc": {
        "arxiv": {
            "en": ["gpt-3", "llama2", "llm-survey", "gemini"],
        },
        "book": {
            "en": [
                "origin-of-species_darwin",
                "a-brief-history-of-time_stephen-hawking"
            ]
        },
        "healthcare": {
            "en": [
                "pubmed_100K-200K_1",
                "pubmed_100K-200K_2",
                "pubmed_100K-200K_3",
                "pubmed_40K-50K_5-merged",
                "pubmed_30K-40K_10-merged"
            ]
        },
        "law": {
            "en": [
                "lex_files_300K-400K",
                "lex_files_400K-500K",
                "lex_files_500K-600K",
                "lex_files_600K-700K"
            ]
        }
    }
}

metric_list = [
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
    "recall_at_10"
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
    "mrr_at_1000"
]


@dataclass
class Benchmark:
    name: str  # [domain]_[language]_[metric], task_key in the json file,
    metric: str  # ndcg_at_1 ,metric_key in the json file
    col_name: str  # [domain]_[language], name to display in the leaderboard
    domain: str
    lang: str
    task: str

qa_benchmark_dict = {}
long_doc_benchmark_dict = {}
for task, domain_dict in dataset_dict.items():
    for domain, lang_dict in domain_dict.items():
        for lang, dataset_list in lang_dict.items():
            if task == "qa":
                benchmark_name = f"{domain}_{lang}"
                benchmark_name = get_safe_name(benchmark_name)
                col_name = f"{domain}_{lang}"
                for metric in dataset_list:
                    qa_benchmark_dict[benchmark_name] = Benchmark(benchmark_name, metric, col_name, domain, lang, task)
            elif task == "long_doc":
                for dataset in dataset_list:
                    col_name = f"{domain}_{lang}_{dataset}"
                    for metric in metric_list:
                        benchmark_name = f"{domain}_{lang}_{dataset}_{metric}"
                        benchmark_name = get_safe_name(benchmark_name)
                        long_doc_benchmark_dict[benchmark_name] = Benchmark(benchmark_name, metric, col_name, domain, lang, task)

BenchmarksQA = Enum('BenchmarksQA', qa_benchmark_dict)
BenchmarksLongDoc = Enum('BenchmarksLongDoc', long_doc_benchmark_dict)

BENCHMARK_COLS_QA = [c.col_name for c in qa_benchmark_dict.values()]

DOMAIN_COLS_QA = list(frozenset([c.domain for c in qa_benchmark_dict.values()]))
LANG_COLS_QA = list(frozenset([c.lang for c in qa_benchmark_dict.values()]))
