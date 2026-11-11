from ..dataset import Connectome, Cached, NameSpaceMap
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Union, Iterable, Type


class Database:
    """Container class for caching large data files."""

    cell_types = Cached("read_csv", ("consolidated_cell_types",))
    connections = Cached("read_csv", ("connections_princeton_no_threshold",))
    classification = Cached("read_csv", ("classification",))
    synapses = Cached("read_csv", ("fafb_v783_princeton_synapse_table",))
    neurons = Cached("read_csv", ("neurons",))

    @classmethod
    def customize(cls, **kwargs):
        kw = {name: cached for (name, cached) in cls.__dict__.items() if isinstance(cached, Cached)}
        for attr, filename in kwargs.items():
            kw[attr] = Cached("read_csv", (filename,))
        return type(cls.__name__, (cls,), kw)

    def __init__(self, directory):
        self.directory = Path(directory)

    def read_csv(self, name):
        df = pd.read_csv(self.directory.joinpath(name).with_suffix(".csv.gz"))
        return df


class FlyWireLocal(Connectome):
    """Class for accessing locally saved FlyWire files.

    Required files downloaded from codex: https://codex.flywire.ai/api/download?dataset=fafb
        - 'consolidated_cell_types.csv.gz'
        - 'connections_no_threshold.csv.gz'
        - 'classification.csv.gz'
        - 'fafb_v783_princeton_synapse_table.csv.gz'
        - 'neurons.csv.gz'
    """

    NAMESPACE = NameSpaceMap(
        cell_id="root_id",
        cell_type="primary_type",
        pre_cell_id="pre_root_id",
        post_cell_id="post_root_id",
    )

    def __init__(self, directory, **kwargs):
        super().__init__("flywire-fafb")
        self.db = Database.customize(**kwargs)(directory)

    @property
    def directory(self):
        return self.db.directory

    def load_cell_types(self) -> dict:
        df = self.db.cell_types
        return {cell_type: root_ids.values for (cell_type, root_ids) in df.groupby("primary_type")["root_id"]}

    def load_brain_regions(self) -> list:
        return self.db.connections["neuropil"].unique().tolist()

    def _get_cell_ids(self, cell_type: str) -> list:
        return self.cell_types[cell_type]

    def _get_id_types(self, cell_ids: Union[int, Iterable[int]]) -> pd.DataFrame:
        cell_ids = np.atleast_1d(cell_ids)
        # Get cell type
        df_type = self.db.cell_types
        df_type = df_type[df_type["root_id"].isin(cell_ids)][["root_id", "primary_type"]]
        # Get side
        df_class = self.db.classification
        df_class = df_class[df_class["root_id"].isin(cell_ids)][["root_id", "side"]]
        return df_type.merge(df_class, on="root_id", how="outer")

    def _get_connections(self, *, pre_ids: Union[int, Iterable[int]]=None, post_ids: Union[int, Iterable[int]]=None,
                        regions: Union[str, Iterable[str]]=None, threshold: int=None) -> pd.DataFrame:
        df = self.db.connections
        if pre_ids is not None:
            pre_ids = np.atleast_1d(pre_ids)
            df = df.query("pre_root_id in @pre_ids")
        if post_ids is not None:
            post_ids = np.atleast_1d(post_ids)
            df = df.query("post_root_id in @post_ids")
        if regions is not None:
            if isinstance(regions, str):
                regions = [regions]
            df = df.query("neuropil in @regions")
        if threshold:
            df = df.query("syn_count >= @threshold")
        df = df.reset_index(drop=True)
        return df[["pre_root_id", "post_root_id", "neuropil", "syn_count"]]

    def _get_neurotransmitter(self, cell_type):
        if cell_type == "LT56":
            return "GLUT"
        neuron_ids = self.get_cell_ids(cell_type)
        neurons = self.db.neurons[self.db.neurons["root_id"].isin(neuron_ids)]
        neurons.loc[pd.isnull(neurons["nt_type"]), "nt_type"] = "UNKNOWN"
        nt_type = neurons["nt_type"].value_counts().idxmax()
        return nt_type

    def _get_synapses(self, *, pre_ids = None, post_ids = None, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Use caveclient instead!")
        # synapses = self.db.synapses
        # print(synapses)
        # try:
        #     if pre_cell_id is None:
        #         assert post_cell_id is not None
        #         return _slice_synapses(synapses, "post_root_id", post_cell_id)
        #     if post_cell_id is None:
        #         assert pre_cell_id is not None
        #         return _slice_synapses(synapses, "pre_root_id", pre_cell_id)
        # except AssertionError:
        #     raise ValueError("At least one of either 'pre_root_id' or 'post_root_id' must be specified.")
        # outputs = _slice_synapses(synapses, "pre_root_id", pre_cell_id)
        # return _slice_synapses(outputs, "post_root_id", post_cell_id)


# def _slice_synapses(df, col, val):
#     start_values = df[col][~pd.isnull(df[col])]
#     start_idxs, = np.where(start_values.values == val)
#     sliced = []
#     for start_idx in start_idxs:
#         try:
#             sliced.append(df.loc[start_values.index[start_idx]:start_values.index[start_idx + 1] - 1])
#         except IndexError:
#             sliced.append(df.loc[start_values.index[start_idx]:])
#     return pd.concat(sliced)
