import argparse
import json
import jsonschema
def gs_imputation(v):
    try:
        vals = json.loads(v)
    except:
         raise argparse.ArgumentTypeError("invalid value \n use -h for help")
    schema = {
        "type" : "object",
        "properties" : {
        "FACTORS" : {"type" : "array","items":{"type": "number"}},
        "EPOCHS" : {"type" : "array","items":{"type": "number"}},
        "NUM_FOLDS":{"type": "number"}
        },
        "required":["FACTORS","EPOCHS"]
    }
    try:
        jsonschema.validate(vals,schema)
    except jsonschema.exceptions.ValidationError:
         raise argparse.ArgumentTypeError("invalid value\n use -h for help ")
    if "NUM_FOLDS" not in vals:
        vals["NUM_FOLDS"] = 2
    return vals

def params_imputation(v):
    try:
        vals = json.loads(v)
    except: 
        raise argparse.ArgumentTypeError("invalid value \n use -h for help")
    
    schema = {
        "type" : "object",
        "properties" : {
        "FACTORS" : {"type" : "number"},
        "EPOCHS" : {"type" : "number"},
        "NUM_FOLDS":{"type": "number"}
        },
        "required":["FACTORS","EPOCHS"]
    }

    try:
        jsonschema.validate(vals,schema)
    except jsonschema.exceptions.ValidationError:
         raise argparse.ArgumentTypeError("invalid value \n use -h for help ")
    if "NUM_FOLDS" not in vals:
        vals["NUM_FOLDS"] = 2
    return vals

def gs_pipelines(v):
    return None