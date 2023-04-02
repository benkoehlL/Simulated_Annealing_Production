import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
import warnings


SCALING_MINMAX1 = "minmax (-1,1)"
SCALING_MINMAX0 = "minmax (0,1)"
SCALING_ZSCORE = 'zscore'


def input_data(filename, sheet_name, scaling="minmax (0,1)", num_jobs=None):

    """
    Returns a dictionary with four problem instances (problem 1,
    ..., 4). Each problem instance consists of a set of jobs 
    characterized by their 'due date', 'setup type' and their 
    processing time for the first ('t_smd') and second production 
    stage ('t_aoi'). Job parameters are further accessible in 
    scaled format.

    ---------------------------------------------------------------
    Parameters:
    ---------------------------------------------------------------
    scaling: Scaling method for input features.
             The following string values are valid:
                - 'minmax (0,1)'
                - 'minmax (-1,1)'
                - 'zscore'

    num_entries: Number of jobs to be imported from each dataset.
    ---------------------------------------------------------------
    """

    warnings.filterwarnings(action="ignore", category=DataConversionWarning)

    feature_columns = ["due date", "family", "t_smd", "t_aoi"]

    dataset = pd.read_excel(filename, sheet_name, nrows=num_jobs)

    if scaling == "minmax (-1,1)":
        features = MinMaxScaler(feature_range=(-1, 1),).fit_transform(
            dataset[feature_columns].values.tolist()
        )
    elif scaling == "zscore":
        features = StandardScaler().fit_transform(
            dataset[feature_columns].values.tolist()
        )
    else:
        features = MinMaxScaler().fit_transform(
            dataset[feature_columns].values.tolist()
        )
    problem = {
         i: {
            "id": i + 1,
            "due date": dataset["due date"][i],
            "family": dataset["family"][i],
            "t_smd": dataset["t_smd"][i],
            "t_aoi": dataset["t_aoi"][i],
            "scaled due date": features[i][0],
            "scaled family": features[i][1],
            "scaled t_smd": features[i][2],
            "scaled t_aoi": features[i][3],
            "alloc to smd": None,
            "alloc to aoi": None,
        }
        for i in range(len(dataset))
    }

    return {"problem": problem, "scaler": scaling}
