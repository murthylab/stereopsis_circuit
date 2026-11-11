from connquer.datasets import Flywire


if __name__ == "__main__":

    # Open Flywire dataset
    dset = Flywire.local()

    # Load column assignment
    df = dset.db.visual_columns

    # Rename columns
    df = df.rename(columns={
        "root_id": "cell_id",
        "type": "cell_type",
        "hemisphere": "side"
    })
    df = df[["cell_id", "cell_type", "side", "column_id", "p", "q"]]
