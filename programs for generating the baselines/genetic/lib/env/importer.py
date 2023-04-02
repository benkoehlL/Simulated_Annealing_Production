import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.exceptions import DataConversionWarning
import os
import warnings


def input_data(directory, scaling="minmax (0,1)", num_jobs=None):

    """
    Returns a dictionary with four problem instances (problem 1,
    ..., 4). Each problem instance consists of a set of jobs 
    characterized by their 'due date, 'setup type' and their 
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

    dataset_1 = pd.read_excel(
        os.path.join(directory, "input_noAOI.xlsx"), sheet_name="dataset 1", nrows=num_jobs
    )
    dataset_2 = pd.read_excel(
        os.path.join(directory, "input_noAOI.xlsx"), sheet_name="dataset 2", nrows=num_jobs
    )
    dataset_3 = pd.read_excel(
        os.path.join(directory, "input_noAOI.xlsx"), sheet_name="dataset 3", nrows=num_jobs
    )
    dataset_4 = pd.read_excel(
        os.path.join(directory, "input_noAOI.xlsx"), sheet_name="dataset 4", nrows=num_jobs
    )

    if scaling == "minmax (-1,1)":
        features_1 = MinMaxScaler(feature_range=(-1, 1),).fit_transform(
            dataset_1[feature_columns].values.tolist()
        )
        features_2 = MinMaxScaler(feature_range=(-1, 1),).fit_transform(
            dataset_2[feature_columns].values.tolist()
        )
        features_3 = MinMaxScaler(feature_range=(-1, 1),).fit_transform(
            dataset_3[feature_columns].values.tolist()
        )
        features_4 = MinMaxScaler(feature_range=(-1, 1),).fit_transform(
            dataset_4[feature_columns].values.tolist()
        )
    elif scaling == "zscore":
        features_1 = StandardScaler().fit_transform(
            dataset_1[feature_columns].values.tolist()
        )
        features_2 = StandardScaler().fit_transform(
            dataset_2[feature_columns].values.tolist()
        )
        features_3 = StandardScaler().fit_transform(
            dataset_3[feature_columns].values.tolist()
        )
        features_4 = StandardScaler().fit_transform(
            dataset_4[feature_columns].values.tolist()
        )
    else:
        features_1 = MinMaxScaler().fit_transform(
            dataset_1[feature_columns].values.tolist()
        )
        features_2 = MinMaxScaler().fit_transform(
            dataset_2[feature_columns].values.tolist()
        )
        features_3 = MinMaxScaler().fit_transform(
            dataset_3[feature_columns].values.tolist()
        )
        features_4 = MinMaxScaler().fit_transform(
            dataset_4[feature_columns].values.tolist()
        )

    problem_1 = {
        i: {
            "id": i + 1,
            "due date": dataset_1["due date"][i],
            "family": dataset_1["family"][i],
            "t_smd": dataset_1["t_smd"][i],
            "t_aoi": dataset_1["t_aoi"][i],
            "scaled due date": features_1[i][0],
            "scaled family": features_1[i][1],
            "scaled t_smd": features_1[i][2],
            "scaled t_aoi": features_1[i][3],
            "alloc to smd": None,
            "alloc to aoi": None,
        }
        for i in range(len(dataset_1))
    }

    problem_2 = {
        i: {
            "id": i + 1,
            "due date": dataset_2["due date"][i],
            "family": dataset_2["family"][i],
            "t_smd": dataset_2["t_smd"][i],
            "t_aoi": dataset_2["t_aoi"][i],
            "scaled due date": features_2[i][0],
            "scaled family": features_2[i][1],
            "scaled t_smd": features_2[i][2],
            "scaled t_aoi": features_2[i][3],
            "alloc to smd": None,
            "alloc to aoi": None,
        }
        for i in range(len(dataset_2))
    }

    problem_3 = {
        i: {
            "id": i + 1,
            "due date": dataset_3["due date"][i],
            "family": dataset_3["family"][i],
            "t_smd": dataset_3["t_smd"][i],
            "t_aoi": dataset_3["t_aoi"][i],
            "scaled due date": features_3[i][0],
            "scaled family": features_3[i][1],
            "scaled t_smd": features_3[i][2],
            "scaled t_aoi": features_3[i][3],
            "alloc to smd": None,
            "alloc to aoi": None,
        }
        for i in range(len(dataset_3))
    }

    problem_4 = {
        i: {
            "id": i + 1,
            "due date": dataset_4["due date"][i],
            "family": dataset_4["family"][i],
            "t_smd": dataset_4["t_smd"][i],
            "t_aoi": dataset_4["t_aoi"][i],
            "scaled due date": features_4[i][0],
            "scaled family": features_4[i][1],
            "scaled t_smd": features_4[i][2],
            "scaled t_aoi": features_4[i][3],
            "alloc to smd": None,
            "alloc to aoi": None,
        }
        for i in range(len(dataset_4))
    }

    problems = {
        "problem 1": problem_1,
        "problem 2": problem_2,
        "problem 3": problem_3,
        "problem 4": problem_4,
    }

    return problems
