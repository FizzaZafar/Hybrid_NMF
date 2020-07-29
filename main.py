import argparse
import argparser
import logging
import gridsearch
import pipeline1
import pipeline2

def parse_args():
    parser = argparse.ArgumentParser(description="hybrid NMF")
    parser.add_argument("--pipeline1_only", type=bool, default=False, 
                    help="generate the imputation and run pipeline1 only")
    parser.add_argument("--grid_search_impute_nmf",type=argparser.gs_imputation, metavar='{"FACTORS":[required],"EPOCHS":[required]}',help="perform a grid search on NMF imputation params")
    parser.add_argument("--impute_params",type= argparser.params_imputation,metavar= '{"FACTORS":900,"EPOCHS":120}',help="only considered if --grid_search_impute is not specified")
    parser.add_argument("--pipeline2_only", type=bool, default=False, 
                    help="run the pipeline2 only. This option should only be used if the imputation was generated already in a previous run")
    parser.add_argument("--grid_search_pipeline2",type=argparser.gs_pipelines, metavar='{"NO_USER_CLUSTERS": [required], "NO_ITEM_CLUSTERS": [required], "GLOBAL_NMF_K": [required], "LOCAL_U_NMF_K": [required], "LOCAL_I_NMF_K": [required], "GLOBAL_NMF_EPOCHS": [required], "LOCAL_U_NMF_EPOCHS": [required], "LOCAL_I_NMF_EPOCHS": [required], "NO_FOLDS":number}',help= "perform a grid search on pipeline params and sets --pipelines_only=True")
    parser.add_argument("--pipeline2_params",type= argparser.params_pipelines,metavar= '{"NO_USER_CLUSTERS": 7, "NO_ITEM_CLUSTERS": 2, "GLOBAL_NMF_K": 4, "LOCAL_U_NMF_K": 30, "LOCAL_I_NMF_K": 30, "GLOBAL_NMF_EPOCHS": 20, "LOCAL_U_NMF_EPOCHS": 8, "LOCAL_I_NMF_EPOCHS": 10}',help="only considered if --grid_search_pipeline2 is not specified")
    parser.add_argument("--gen_submission",type=bool,default=True,help="generate the submission csv")
    parser.add_argument("--hold_out_percentage", type=int, choices=range(0, 101), metavar="[0-100]", help="percentage of values to be held out while training", default=0)
    args = parser.parse_args()
    print(args)
    return args

def run(args):
    if args.grid_search_impute_nmf != None:
        gridsearch.impute(args.grid_search_impute_nmf)
        return
    elif args.pipeline1_only:
        only_pipeline1(args)
        return

    if args.grid_search_pipeline2 != None:
        gridsearch.pip2(grid_search_pipeline2)
    elif args.pipeline2_only:
        only_pipeline2(args)
        return
    
    else:
        logging.info("Running both imputation and pipelines")
        if args.impute_params == None:
            logging.info("Running imputation with default params")
            args.impute_params = argparser.params_imputation("{}")
        else:
            logging.info("Running imputation with params: "+str(args.impute_params))
        
        if args.pipeline2_params == None:
            logging.info("Running pipelines with default params")
            args.pipeline2_params = argparser.params_pipelines("{}")
        else:
            logging.info("Running pipelines with params: "+str(args.pipeline2_params))


def only_pipeline1(args):
    if args.impute_params == None:
        logging.info("Running imputation with default params")
        args.impute_params = argparser.params_imputation("{}")
    else:
        logging.info("Running imputation with params: "+str(args.impute_params))
    pipeline1.do(args.impute_params)

def only_pipeline2(args):
    if args.pipeline2_params == None:
        logging.info("Running pipelines with default params")
        args.pipeline2_params = argparser.params_pipelines("{}")
    else:
        logging.info("Running pipelines with params: "+str(args.pipeline2_params))
    pipeline2.do(args.pipeline2_params,args.gen_submission)
    

def main():
    logging.basicConfig(filename="log.log", filemode='w', level=logging.INFO)
    args = parse_args()
    run(args)
    

if __name__ == "__main__":
    main()


