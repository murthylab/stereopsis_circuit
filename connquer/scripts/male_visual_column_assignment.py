from connquer.datasets import Neuprint
from neuprint import fetch_neurons, NeuronCriteria as NC
import pandas as pd
import numpy as np


def assign_column_ids(df_side):

    # Create copy
    df_copy = df_side.copy()

    # Get unique pq coordinates
    pq = np.unique(df_copy[["p", "q"]], axis=0)

    # Center pq coordinate system
    origin = np.round(np.mean(pq, axis=0)).astype("i4")
    pq = pq - origin
    df_copy.loc[:, ["p", "q"]] = df_copy[["p", "q"]] - origin

    # Create a mapping between coordinates and column ids
    pqi = np.column_stack((pq, np.arange(len(pq)) + 1))
    pq_map = np.zeros((len(np.unique(pq[:, 0])), len(np.unique(pq[:, 1]))), dtype="i4")
    pq_map[pqi[:, 0] - 1, pqi[:, 1] - 1] = pqi[:, 2]

    # Set column id
    df_copy.loc[:, ["column_id"]] = pq_map[df_copy["p"].values - 1, df_copy["q"].values - 1]

    return df_copy


columnar_types = (
    "L1", "L2", "L5", "C3", "Mi1", "Mi4", "Mi9", "T1", "Tm1", "Tm2", "Tm9", "Tm20"
)


if __name__ == "__main__":

    # Open male CNS dataset
    dset = Neuprint.male_cns()

    df = []
    # Get cell body info from neuprint server for each columnar type
    for cell_type in columnar_types:
        print(cell_type)
        cell_ids = dset.get_cell_ids(cell_type)
        neuron_df, _ = fetch_neurons(NC(bodyId=[int(cell_id) for cell_id in cell_ids]))
        column_info = neuron_df[["bodyId", "somaSide", "type", "assignedOlHex1", "assignedOlHex2"]]
        # Remove nans
        column_info = column_info[(~pd.isnull(column_info["assignedOlHex1"]) & (~pd.isnull(column_info["assignedOlHex2"])))]
        df.append(column_info)
    # Concatenate info for all cell types into single DataFrame
    df = pd.concat(df, ignore_index=True)
    # Change formatting
    df[["somaSide", "assignedOlHex1", "assignedOlHex2"]] = df.transform({
        "somaSide": lambda x: dict(L="left", R="right")[x],
        "assignedOlHex1": int,
        "assignedOlHex2": int
    })
    # Rename columns
    df = df.rename(columns={
        "bodyId": "cell_id",
        "type": "cell_type",
        "somaSide": "side",
        "assignedOlHex1": "q",
        "assignedOlHex2": "p"
    })
    # Assign column ids
    df_r = assign_column_ids(df.query("side == 'right'"))
    df_l = assign_column_ids(df.query("side == 'left'"))
    df = pd.concat([df_r, df_l], ignore_index=True)
    df = df[["cell_id", "cell_type", "side", "column_id", "p", "q"]]
