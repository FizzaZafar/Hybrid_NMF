
# Baselines
 Different existing collaborative filtering algorithms are considered and evaluated on the data set

## Running baselines
### surprise
To run the algorithms evaluated using the surprise library run `python3 surprise/bench_surprise`. This generates `surprise/summary.txt` with the validation RMSE for each of the algorithms evaluated

### fastai
To run neural networks developed using fastai package run `python3 fastai/neural_networks.py`. This generates `fastai/history.csv` with the RMSEs. 

### autorec
To run the Autorec algorithm on the given data set run `python3 autorec/main.py`. This generates files in `autorec/results/` with train and validation RMSE for each epoch.

## Evaluation
| Algorithm       | Validation RMSE | Test RMSE | Parameters                                   |
|-----------------|-----------------|-----------|----------------------------------------------|
| NMF             | 1.0714975       | 1.18537   | epochs = 200, factors = 200                  |
| SVDpp           | 1.0714975       | 1.23459   | epochs = 200, factors = 200                  |
| Neural Network^ | 1.007911        | 0.99635   | layers=[200,128,64,16]                       |
| AutoRec*         | 1.3462          | 1.39741   | epochs=1, hidden neurons=100, batch size=100 |
| Baseline        | 0.9996473       | 0.99768   | method=ALS, epochs = 10                      |
| Random          | 1.48162         | 1.48097   | method=ALS, epochs = 10                      |
| KNN             | 1.00912         | 1.22238   | K = 40                                       |

\*AutoRec was run using the [implementation](https://github.com/gtshs2/Autorec)

\^Neural Networks were evaluated using [fastai](https://pypi.org/project/fastai/) package

All other algorithms were evaluated using the [surprise](http://surpriselib.com/) library