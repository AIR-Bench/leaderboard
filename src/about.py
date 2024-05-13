# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">AIR-Bench</h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
"""

# Which evaluations are you running? how can people reproduce what you have?
BENCHMARKS_TEXT = f"""
## How it works

Check more information at [our GitHub repo](https://github.com/AIR-Bench/AIR-Bench)
"""

EVALUATION_QUEUE_TEXT = """
## Steps for submit to AIR-Bench

1. Install AIR-Bench
```bash
# Clone the repo
git clone https://github.com/AIR-Bench/AIR-Bench.git

# Install the package
cd AIR-Bench
pip install .
```
2. Run the evaluation script
```bash
cd AIR-Bench/scripts
# Run all tasks
python run_AIR-Bench.py \
--output_dir ./search_results \
--encoder BAAI/bge-m3 \
--encoder_link https://huggingface.co/BAAI/bge-m3 \
--reranker BAAI/bge-reranker-v2-m3 \
--reranker_link https://huggingface.co/BAAI/bge-reranker-v2-m3 \
--search_top_k 1000 \
--rerank_top_k 100 \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 512 \
--pooling_method cls \
--normalize_embeddings True \
--use_fp16 True \
--add_instruction False \
--overwrite False

# Run the tasks in the specified task type
python run_AIR-Bench.py \
--task_types long-doc \
--output_dir ./search_results \
--encoder BAAI/bge-m3 \
--encoder_link https://huggingface.co/BAAI/bge-m3 \
--reranker BAAI/bge-reranker-v2-m3 \
--reranker_link https://huggingface.co/BAAI/bge-reranker-v2-m3 \
--search_top_k 1000 \
--rerank_top_k 100 \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 512 \
--pooling_method cls \
--normalize_embeddings True \
--use_fp16 True \
--add_instruction False \
--overwrite False

# Run the tasks in the specified task type and domains
python run_AIR-Bench.py \
--task_types long-doc \
--domains arxiv book \
--output_dir ./search_results \
--encoder BAAI/bge-m3 \
--encoder_link https://huggingface.co/BAAI/bge-m3 \
--reranker BAAI/bge-reranker-v2-m3 \
--reranker_link https://huggingface.co/BAAI/bge-reranker-v2-m3 \
--search_top_k 1000 \
--rerank_top_k 100 \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 512 \
--pooling_method cls \
--normalize_embeddings True \
--use_fp16 True \
--add_instruction False \
--overwrite False

# Run the tasks in the specified languages
python run_AIR-Bench.py \
--languages en \
--output_dir ./search_results \
--encoder BAAI/bge-m3 \
--encoder_link https://huggingface.co/BAAI/bge-m3 \
--reranker BAAI/bge-reranker-v2-m3 \
--reranker_link https://huggingface.co/BAAI/bge-reranker-v2-m3 \
--search_top_k 1000 \
--rerank_top_k 100 \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 512 \
--pooling_method cls \
--normalize_embeddings True \
--use_fp16 True \
--add_instruction False \
--overwrite False

# Run the tasks in the specified task type, domains, and languages
python run_AIR-Bench.py \
--task_types qa \
--domains wiki web \
--languages en \
--output_dir ./search_results \
--encoder BAAI/bge-m3 \
--encoder_link https://huggingface.co/BAAI/bge-m3 \
--reranker BAAI/bge-reranker-v2-m3 \
--reranker_link https://huggingface.co/BAAI/bge-reranker-v2-m3 \
--search_top_k 1000 \
--rerank_top_k 100 \
--max_query_length 512 \
--max_passage_length 512 \
--batch_size 512 \
--pooling_method cls \
--normalize_embeddings True \
--use_fp16 True \
--add_instruction False \
--overwrite False
```
3. Package the search results.
```bash
python zip_results.py \
--results_path search_results/bge-m3 \
--save_path search_results/zipped_results
```
4. Upload the `.zip` file on this page and fill in the model information. 
5. Congratulation! Your results will be shown on the leaderboard in up to one hour.
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
"""
