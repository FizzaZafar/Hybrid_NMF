# hybrid_nmf
## Quickstart
run `python3 main.py` to run the algorithm with default parameter values. 
This generates `results/submission.csv` 

## Overview
### Collaborative filtering
Collaborative   filtering is   a   technique   used   by recommendation  systems,  wherein  ratings  of  other  users  and items  are  used  to  predict  the  rating  for  a  given  (user,  item) pair. Popular  collaborative  filtering  algorithms use  matrix  factorization  and  neighborhood  based  methods.

### Our approach
 We  have developed  a  novel  approach  that  combines  ideas  of  both factorization  and  neighborhood  based  methods.  We  obtain ratings  by  considering  hidden  factors  for  the  overall  ratings matrix  as  well  as  sub-matrices  obtained  by  clustering  users and items.

## Dataset
This project was used for in-class kaggle [competition](https://www.kaggle.com/c/cil-collab-filtering-2020/). Data about the users and items can be found in the [repo](https://github.com/suprajasridhara/hybrid_nmf/blob/master/data/data_train_clean.csv) or on [kaggle](https://www.kaggle.com/c/cil-collab-filtering-2020/data).

## Baselines
The following existing approaches to collaborative filtering were evaluated. The table below enumerates their validation and test RMSE.

More information can be found in [baselines](https://github.com/suprajasridhara/hybrid_nmf/tree/master/baselines)

## More Info
For a more detailed discussion about the method and the evaluation steps used see the [report](https://TODO:add_link_to_report)


