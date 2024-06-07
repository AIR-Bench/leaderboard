# Your leaderboard name
TITLE = """<h1 align="center" id="space-title">AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark
 (v0.0.3) </h1>"""

# What does your leaderboard evaluate?
INTRODUCTION_TEXT = """
## Check more information at [our GitHub repo](https://github.com/AIR-Bench/AIR-Bench)
"""

# Which evaluations are you running? how can people reproduce what you have?
BENCHMARKS_TEXT = f"""
## How the test data are generated?
### Find more information at [our GitHub repo](https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/data_generation.md)

## FAQ
- Q: Will you release a new version of datasets regularly? How often will AIR-Bench release a new version?
  - A: Yes, we plan to release new datasets on regular basis. However, the update frequency is to be decided.  

- Q: As you are using models to do the quality control when generating the data, is it biased to the models that are used?
  - A: Yes, the results is biased to the chosen models. However, we believe the datasets labeled by human are also biased to the human's preference. The key point to verify is whether the model's bias is consistent with the human's. We use our approach to generate test data using the well established MSMARCO datasets. We benchmark different models' performances using the generated dataset and the human-label DEV dataset. Comparing the ranking of different models on these two datasets, we observe the spearman correlation between them is 0.8211 (p-value=5e-5). This indicates that the models' perference is well aligned with the human. Please refer to [here](https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/available_evaluation_results.md#consistency-with-ms-marco) for details

"""

EVALUATION_QUEUE_TEXT = """
## Check out the submission steps at [our GitHub repo](https://github.com/AIR-Bench/AIR-Bench/blob/main/docs/submit_to_leaderboard.md)
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
"""
