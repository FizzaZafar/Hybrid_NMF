import argparse
import argparser


def main():
    parser = argparse.ArgumentParser(description="hybrid NMF")
    parser.add_argument("--impute_only", type=bool, default=False, 
                    help="generate the imputation only")
    parser.add_argument("--grid_search_impute",type=argparser.gs_imputation, metavar='{"FACTORS":[required],"EPOCHS":[required],"NUM_FOLDS":2}', default={},help="this argument performs a grid search on imputation params and sets --impute_only=True")
    parser.add_argument("--impute_params",type= argparser.params_imputation,metavar= '{"FACTORS":number_required,"EPOCHS":number_required,"NUM_FOLDS",2}',help="this param is only considered if --grid_search_impute=False")
    parser.add_argument("--pipelines_only", type=bool, default=False, 
                    help="run the pipelines only. This option should only be used if the impute only option was used to generate the imputation earlier")
    parser.add_argument("--grid_search_pipelines",type=argparser.gs_pipelines, metavar='{"FACTORS":[],"EPOCHS":[]}', default=False,help="this argument performs a grid search on imputation params and sets --impute_only=True")

    parser.add_argument("--gen_submission",type=bool,default=True,help="generate the submission csv")
    parser.add_argument("--validate", type=bool,default=True, help="write validation rmse to file after training model")
    parser.add_argument("--hold_out_percentage", type=int, choices=range(0, 101), metavar="[0-100]", help="percentage of values to be held out while training", default=0)
    args = parser.parse_args()
    print(args)

    

if __name__ == "__main__":
    main()


