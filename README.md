# Hybrid_NMF

## Dependencies
The project depends on the following packages to run

numpy: `pip3 install numpy`

pandas: `pip3 install pandas`

surprise: `pip3 install surprise`

sklearn: `pip3 install sklearn`

fastai: `pip3 install fastai`

tensorflow: `pip3 install tensorflow`

seaborn: `pip3 install seaborn`

matplotlib: `pip3 install matplotlib`

## Quickstart
run `python3 src/main.py` to run the algorithm with default parameter values. 
This generates `results/submission.csv` 

## Overview
### Collaborative filtering
Collaborative   filtering is   a   technique   used   by recommendation  systems,  wherein  ratings  of  other  users  and items  are  used  to  predict  the  rating  for  a  given  (user,  item) pair. Popular  collaborative  filtering  algorithms use  matrix  factorization  and  neighborhood  based  methods.

### Approach
 This repo contains a novel  approach  that  combines  ideas  of  both factorization  and  neighborhood  based  methods.  It  obtains ratings  by  considering  hidden  factors  for  the  overall  ratings matrix  as  well  as  sub-matrices  obtained  by  clustering  users and items.

## Dataset
This project was used for in-class kaggle [competition](https://www.kaggle.com/c/cil-collab-filtering-2020/). Data about the users and items can be found in the [repo](https://github.com/suprajasridhara/hybrid_nmf/blob/master/data/data_train_clean.csv) or on [kaggle](https://www.kaggle.com/c/cil-collab-filtering-2020/data).

## Baselines
Different approaches to collaborative filtering were evaluated. More information can be found in [baselines](https://github.com/suprajasridhara/hybrid_nmf/tree/master/baselines)

## Implementation
The implementation of the proposed method can be found in [/src](https://github.com/suprajasridhara/hybrid_nmf/tree/master/src). The command line interface supports different arguments. For more details run `python3 scr/main.py -h` or see [README](https://github.com/suprajasridhara/hybrid_nmf/blob/master/src/README.md)

## Plots
Different plots for the baselines and our implementation can be generated from the `/plots` folders. For more details see [README](https://github.com/suprajasridhara/hybrid_nmf/blob/master/plots/README.md)

## More Info
For a more detailed discussion about the method and the evaluation steps used see the [report](https://TODO:add_link_to_report)

## Authors
Fizza Zafar (fzafar@ethz.ch) and Supraja Sridhara(ssridhara@ethz.ch)