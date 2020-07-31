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
        },
        "required":["FACTORS","EPOCHS"]
    }
    try:
        jsonschema.validate(vals,schema)
    except jsonschema.exceptions.ValidationError:
         raise argparse.ArgumentTypeError("invalid value\n use -h for help ")

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
        "EPOCHS" : {"type" : "number"}
        }
    }

    try:
        jsonschema.validate(vals,schema)
    except jsonschema.exceptions.ValidationError:
         raise argparse.ArgumentTypeError("invalid value \n use -h for help ")

    if "FACTORS" not in vals:
        vals["FACTORS"] = 900
    if "EPOCHS" not in vals:
        vals["EPOCHS"] = 175
    return vals

def gs_pipelines(v):
    print(v)
    try:
        vals = json.loads(v)
    except: 
        raise argparse.ArgumentTypeError("invalid value \n use -h for help")

    schema = {
        "type" : "object",
        "properties" : {
            "NO_USER_CLUSTERS" : {"type" : "array","items":{"type": "number"}},
            "NO_ITEM_CLUSTERS" : {"type" : "array","items":{"type": "number"}},
            "LOCAL_U_NMF_K" : {"type" : "array","items":{"type": "number"}},
            "LOCAL_I_NMF_K" : {"type" : "array","items":{"type": "number"}},
            "LOCAL_U_NMF_EPOCHS" : {"type" : "array","items":{"type": "number"}},
            "LOCAL_I_NMF_EPOCHS" : {"type" : "array","items":{"type": "number"}},
            "NO_FOLDS" : {"type": "number"},
        },
        "required":["NO_USER_CLUSTERS","NO_ITEM_CLUSTERS","LOCAL_U_NMF_K","LOCAL_I_NMF_K","LOCAL_U_NMF_EPOCHS","LOCAL_I_NMF_EPOCHS"]
    }
    
    try:
        jsonschema.validate(vals,schema)
    except jsonschema.exceptions.ValidationError:
         raise argparse.ArgumentTypeError("invalid value \n use -h for help ")
    if "NO_FOLDS" not in vals:
        vals["NO_FOLDS"] = 2
    return vals

def params_pipelines(v):
    try:
        vals = json.loads(v)
    except: 
        raise argparse.ArgumentTypeError("invalid value \n use -h for help")

    schema = {
        "type" : "object",
        "properties" : {
            "NO_USER_CLUSTERS" : {"type" : "number"},
            "NO_ITEM_CLUSTERS" : {"type" : "number"},
            "LOCAL_U_NMF_K" : {"type" : "number"},
            "LOCAL_I_NMF_K" : {"type" : "number"},
            "LOCAL_U_NMF_EPOCHS" : {"type" : "number"},
            "LOCAL_I_NMF_EPOCHS" : {"type" : "number"},
            "NO_FOLDS" : {"type": "number"},
        },
    }

    vals["GLOBAL_NMF_K"] = 4
    vals["GLOBAL_NMF_EPOCHS"] = 20

    if "NO_USER_CLUSTERS" not in vals:
        vals["NO_USER_CLUSTERS"] = 7
    if "NO_ITEM_CLUSTERS" not in vals:
        vals["NO_ITEM_CLUSTERS"] = 2
    if "LOCAL_U_NMF_K" not in vals:
        vals["LOCAL_U_NMF_K"] = 30
    if "LOCAL_I_NMF_K" not in vals:
        vals["LOCAL_I_NMF_K"] = 30
    if "LOCAL_U_NMF_EPOCHS" not in vals:
        vals["LOCAL_U_NMF_EPOCHS"] = 8
    if "LOCAL_I_NMF_EPOCHS" not in vals:
        vals["LOCAL_I_NMF_EPOCHS"] = 10
    if "NO_FOLDS" not in vals:
        vals["NO_FOLDS"] = 2    
    
    return vals

    
