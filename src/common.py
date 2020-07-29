
import pandas as pd
def read_data(train_file, test_file): 
    full_data = pd.read_csv(train_file)
    data_sub = pd.read_csv(test_file)
    full_data["User"] = full_data["User"]-1
    full_data["Movie"] = full_data["Movie"]-1
    data_sub["User"] = data_sub["User"]-1
    data_sub["Movie"] = data_sub["Movie"]-1
    return full_data, data_sub