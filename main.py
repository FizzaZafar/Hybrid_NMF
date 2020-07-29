import argparse
import argparser
import logging
import gridsearch

def parse_args():
    parser = argparse.ArgumentParser(description="hybrid NMF")
    parser.add_argument("--impute_only", type=bool, default=False, 
                    help="generate the imputation only")
    parser.add_argument("--grid_search_impute_nmf",type=argparser.gs_imputation, metavar='{"FACTORS":[required],"EPOCHS":[required]}', default={},help="perform a grid search on NMF imputation params")
    parser.add_argument("--impute_params",type= argparser.params_imputation,metavar= '{"FACTORS":900,"EPOCHS":120,"NO_FOLDS",2}',help="only considered if --grid_search_impute is not specified")
    parser.add_argument("--pipelines_only", type=bool, default=False, 
                    help="run the pipelines only. This option should only be used if the impute only option was used to generate the imputation earlier")
    parser.add_argument("--grid_search_pipelines",type=argparser.gs_pipelines, metavar='{"NO_USER_CLUSTERS": [required], "NO_ITEM_CLUSTERS": [required], "GLOBAL_NMF_K": [required], "LOCAL_U_NMF_K": [required], "LOCAL_I_NMF_K": [required], "GLOBAL_NMF_EPOCHS": [required], "LOCAL_U_NMF_EPOCHS": [required], "LOCAL_I_NMF_EPOCHS": [required], "NO_FOLDS":number}', default=False,help= "perform a grid search on pipeline params and sets --pipelines_only=True")
    parser.add_argument("--pipelines_params",type= argparser.params_pipelines,metavar= '{"FACTORS":number_required,"EPOCHS":number_required,"NO_FOLDS",2}',help="only considered if --grid_search_impute is not specified")
    parser.add_argument("--gen_submission",type=bool,default=True,help="generate the submission csv")
    parser.add_argument("--validate", type=bool,default=True, help="write validation rmse to file after training model")
    parser.add_argument("--hold_out_percentage", type=int, choices=range(0, 101), metavar="[0-100]", help="percentage of values to be held out while training", default=0)
    args = parser.parse_args()
    print(args)
    return args

def run(args):
    if args.grid_search_impute_nmf != None:
        gridsearch.impute(args.grid_search_impute_nmf)
    elif args.impute_only:
        only_impute()

    if args.grid_search_pipelines != None:
        logging.info("Running grid search for pipelines")
    elif args.pipelines_only:
        only_pipeline()
    
    else:
        logging.info("Running both imputation and pipelines")
        if args.impute_params == None:
            logging.info("Running imputation with default params")
            args.impute_params = argparser.params_imputation("{}")
        else:
            logging.info("Running imputation with params: "+str(args.impute_params))
        
        if args.pipelines_params == None:
            logging.info("Running pipelines with default params")
            args.pipelines_params = argparser.params_pipelines("{}")
        else:
            logging.info("Running pipelines with params: "+str(args.pipeline_params))


def only_impute():
    if args.impute_params == None:
            logging.info("Running imputation with default params")
            args.impute_params = argparser.params_imputation("{}")
        else:
            logging.info("Running imputation with params: "+str(args.impute_params))

def only_pipeline():
    if args.pipelines_params == None:
            logging.info("Running pipelines with default params")
            args.pipelines_params = argparser.params_pipelines("{}")
        else:
            logging.info("Running pipelines with params: "+str(args.pipeline_params))
def main():
    logging.basicConfig(filename="log.log", filemode='w', level=logging.INFO)

    args = parse_args()
    run(args)
    

if __name__ == "__main__":
    main()


