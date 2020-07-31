# Implementation

Run `python3 main.py` to generate the sumbmission file in `../results` with default parameters 

## Command line arguments
run `python3 main.py -h` to list all available command line arguments 

```
usage: main.py [-h] [--pipeline1_only PIPELINE1_ONLY]
               [--grid_search_impute_nmf {"FACTORS":[required],"EPOCHS":[required]}]
               [--impute_params {"FACTORS":900,"EPOCHS":120}]
               [--pipeline2_only PIPELINE2_ONLY]
               [--grid_search_pipeline2 {"NO_USER_CLUSTERS": [required], "NO_ITEM_CLUSTERS": [required], "LOCAL_U_NMF_K": [required], "LOCAL_I_NMF_K": [required],  "LOCAL_U_NMF_EPOCHS": [required], "LOCAL_I_NMF_EPOCHS": [required], "NO_FOLDS":number}]
               [--pipeline2_params {"NO_USER_CLUSTERS": 7, "NO_ITEM_CLUSTERS": 2, "LOCAL_U_NMF_K": 30, "LOCAL_I_NMF_K": 30, "LOCAL_U_NMF_EPOCHS": 8, "LOCAL_I_NMF_EPOCHS": 10}]
               [--gen_submission GEN_SUBMISSION]
               [--blending_model {LinearRegression,LassoCV,ElasticNetCV,StackingRegressor,RidgeCV,SGDRegressor,Perceptron}]

hybrid NMF

optional arguments:
  -h, --help            show this help message and exit
  --pipeline1_only PIPELINE1_ONLY
                        generate the imputation and run pipeline1 only
  --grid_search_impute_nmf {"FACTORS":[required],"EPOCHS":[required]}
                        perform a grid search on NMF imputation params
  --impute_params {"FACTORS":900,"EPOCHS":120}
                        only considered if --grid_search_impute is not
                        specified
  --pipeline2_only PIPELINE2_ONLY
                        run the pipeline2 only. This option should only be
                        used if the imputation was generated already in a
                        previous run
  --grid_search_pipeline2 {"NO_USER_CLUSTERS": [required], "NO_ITEM_CLUSTERS": [required], "LOCAL_U_NMF_K": [required], "LOCAL_I_NMF_K": [required],  "LOCAL_U_NMF_EPOCHS": [required], "LOCAL_I_NMF_EPOCHS": [required], "NO_FOLDS":number}
                        perform a grid search on pipeline params and sets
                        --pipelines_only=True
  --pipeline2_params {"NO_USER_CLUSTERS": 7, "NO_ITEM_CLUSTERS": 2, "LOCAL_U_NMF_K": 30, "LOCAL_I_NMF_K": 30, "LOCAL_U_NMF_EPOCHS": 8, "LOCAL_I_NMF_EPOCHS": 10}
                        only considered if --grid_search_pipeline2 is not
                        specified
  --gen_submission GEN_SUBMISSION
                        generate the submission csv
  --blending_model {LinearRegression,LassoCV,ElasticNetCV,StackingRegressor,RidgeCV,SGDRegressor,Perceptron}
                        blending models

```
### Notes

```
The parameters [--grid_search_impute_nmf, --pipeline1_only, --grid_search_pipeline2, --pipeline2_only] 
are evaluated in order. Only one option can be run at a time. 
--impute_params is considered only when running with 
--pipeline1_only or default (both pipelines)
--pipeline2_params is considered only when running with --pipeline2_only or default (both pipelines)
--blending_model uses ElasticNetCV by default
--gen_submission is True by default and generates the submission file when --pipeline2_only or default (both pipelines) is run

```
## Grid search
### Imputation
Run `python3 main --grid_search_impute_nmf {"FACTORS":[required],"EPOCHS":[required]}`
This generates `results/gs_impute_results.csv` with RMSE values.

The grid search for the imputation is always run with 5 fold cross validation. 

### Pipeline2
Run `python3 main --grid_search_impute_nmf {"NO_USER_CLUSTERS": [required], "NO_ITEM_CLUSTERS": [required], "LOCAL_U_NMF_K": [required], "LOCAL_I_NMF_K": [required],  "LOCAL_U_NMF_EPOCHS": [required], "LOCAL_I_NMF_EPOCHS": [required], "NO_FOLDS":number}`
This generates `results/csv.json` with RMSE values.

`NO_FOLDS` is optional, default value is 2 i.e. by default the grid search is run with 2 fold cross validation

