# Course workload analytics

Supplementary repository for peer review of manuscript "Insights into undergraduate pathways using course load analytics" submitted to LAK '23.

## Folder structure

* `utils.py`: Collection of functions and classes used throughout scripts.
* `1-create-course-features.py`: Script to create LMS and enrollment-based variables for individual semesters and save results to `.csv` files. Run via `python3 1-create-course-features.py '2021 Spring'` or any other semester between Spring '17 and Spring '21.
* `2-prepare-training-data.py`: Script to pre-process survey data of Spring '21 semester and join course-level variables created via `1-create-course-features.py` to the survey data.
* `3-train-model.py`: Script to perform cross-validated random search for a set of models specified in `run_model_training()` and a specified set of target variables (`LABELS`) in `utils.py`. All relevant results are saved as pickle objects to the `models/` folder.
* `4-evaluate-models.ipynb`: Notebook to export the cross-validation error and test error and perform robustness checks for predictions with high discrepancy to credit hour designation. 
* `5-extrapolate.ipynb`: Notebook to cread course-level course load predictions for all previous semesters based on LMS and enrollment data for a specified set of models and target variables.
* `6-analysis.ipynb`: Notebook to perform descriptive and inferential analyses on the scaled course load predictions on a course and student-semester level. Includes code to reproduce plots in manuscript.
* `7-modeling.R`: R code to perform likelihood-ratio tests for model selection on logistic regression models of dropout and on-time graduation.

