|Validation RMSE | Test RMSE | Parameters |
| ------------- | ------------- | ----- | ---- |
|NMF                | 1.0714975                | 1.18537            | epochs = 200, |factors = 200 |
|SVDpp              | 1.0168                   | 1.23459            | epochs = 200, factors = 200 |
|Neural Network^     | 1.007911                 | 0.99635            | layers={[200,128,64,16]}  |
|AutoRec*            | 1.3462                   | 1.39741            | epochs=1, hidden neurons=100, batch size=100 |
|Baseline           | 0.9996473                | 0.99768            | method=ALS, epochs = 10     |
|Random             | 1.48162                  | 1.48097            | method=ALS, epochs = 10     |
|KNN                | 1.00912                  | 1.22238            | K = 40|

\*AutoRec was run using the default [implementation] 

\^dsf

All other algorithms were evaluated using the [surprise] library