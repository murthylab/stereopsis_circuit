from ..dataset import *
from neuprint import (Client as neuClient,
                      NeuronCriteria as NC,
                      SynapseCriteria as SC,
                      fetch_neurons,
                      fetch_simple_connections,
                      fetch_adjacencies,
                      fetch_synapses,
                      fetch_synapse_connections)
from neuprint.queries import fetch_roi_hierarchy
import pandas as pd
import numpy as np
from typing import Union, Iterable
pd.options.mode.chained_assignment = None


def recursive_dict(d):
    end_nodes = []
    for k, val in d.items():
        if isinstance(val, dict) and len(val) > 0:
            end_nodes.extend(recursive_dict(val))
        else:
            end_nodes.append(k)
    return end_nodes


def subdict(d, keys):
    return {k: d[k] for k in keys if k in d}


class NeuprintClient(Connectome, ClientMixin):
    """Class for accessing neuprint datasets."""

    _server = 'neuprint.janelia.org'

    roi_hierarchy = Cached("fetch_roi_hierarchy")

    NAMESPACE = NameSpaceMap(
        cell_type="type",
        cell_id="bodyId",
        pre_cell_id="bodyId_pre",
        post_cell_id="bodyId_post",
        neuropil="roi",
        syn_count="weight",
        x="x_post",
        y="y_post",
        z="z_post"
    )

    NEUROTRANSMITTER_MAP = {
        "acetylcholine": "ACH",
        "gaba": "GABA",
        "glutamate": "GLUT"
    }

    @staticmethod
    def fetch_roi_hierarchy():
        return fetch_roi_hierarchy(include_subprimary=False)

    def connect_to_client(self, token, dataset):
        return neuClient(self._server, dataset=dataset, token=token)

    def load_cell_types(self) -> dict:
        neuron_df, conn_df = fetch_neurons(NC())
        return {cell_type: ids.values for (cell_type, ids) in neuron_df.groupby("type")["bodyId"]}

    def load_brain_regions(self) -> list:
        return [s.strip("*") for s in recursive_dict(self.roi_hierarchy)]

    def _get_cell_ids(self, cell_type: str) -> list:
        neuron_df, conn_df = fetch_neurons(NC(type=cell_type))
        return neuron_df["bodyId"].values

    def _get_id_types(self, cell_ids: Union[int, Iterable[int]]) -> pd.DataFrame:
        cell_ids = np.unique(np.atleast_1d(cell_ids))
        if not len(cell_ids):
            return pd.DataFrame(columns=["bodyId", "type", "side"])
        neuron_df, _ = fetch_neurons(NC(bodyId=[int(cell_id) for cell_id in cell_ids]))
        df = neuron_df[["bodyId", "type"]]
        neuron_df.loc[pd.isnull(neuron_df["instance"]), "instance"] = "X"
        df["side"] = neuron_df["instance"].map(lambda x: {"R": "right", "L": "left"}.get(x[-1], None))
        return df

    def _get_connections(self, *, pre_ids: Union[int, Iterable[int]] = None, post_ids: Union[int, Iterable[int]] = None,
                         regions: Union[str, Iterable[str]] = None, threshold: int = None) -> pd.DataFrame:
        if pre_ids is not None and len(np.atleast_1d(pre_ids)):
            pre_ids = NC(bodyId=[int(cell_id) for cell_id in np.atleast_1d(pre_ids)])
        if post_ids is not None and len(np.atleast_1d(post_ids)):
            post_ids = NC(bodyId=[int(cell_id) for cell_id in np.atleast_1d(post_ids)])
        try:
            _, df = fetch_adjacencies(sources=pre_ids, targets=post_ids, rois=regions, min_roi_weight=threshold)
            df = df.reset_index(drop=True)
            df = df[["bodyId_pre", "bodyId_post", "roi", "weight"]]
        except RuntimeError as err:
            msg, *_ = err.args
            if msg == "No neurons match your target criteria":
                df = pd.DataFrame(columns=["bodyId_pre", "bodyId_post", "roi", "weight"], dtype=int)
            else:
                raise
        return df

    @staticmethod
    def _fetch_synapse_connections(pre_cell_ids, post_cell_ids, neuropil):
        dfs = []
        for pre_cell_id in pre_cell_ids:
            for post_cell_id in post_cell_ids:
                df = fetch_synapse_connections(source_criteria=NC(bodyId=pre_cell_id),
                                               target_criteria=NC(bodyId=post_cell_id),
                                               synapse_criteria=SC(rois=neuropil))
                dfs.append(df)
        return pd.concat(dfs, axis=0)

    @staticmethod
    def _fetch_synapses(cell_ids, neuropil, syn_type):
        dfs = []
        for cell_id in cell_ids:
            df = fetch_synapses(cell_id, synapse_criteria=SC(rois=neuropil, type=syn_type))
            dfs.append(df)
        return pd.concat(dfs)

    def _get_neurotransmitter(self, cell_type):
        neurons, _ = fetch_neurons(NC(type=cell_type))
        nt_type = neurons["consensusNt"].value_counts().idxmax()
        return self.NEUROTRANSMITTER_MAP.get(nt_type, nt_type)

    def _get_synapses(self, *, pre_cell_id: int = None, post_cell_id: int = None, neuropil: str = None) -> pd.DataFrame:
        raise NotImplementedError
        # if pre_cell_id is None and post_cell_id is None:
        #     raise ValueError("pre_cell_id and/or post_cell_id must be specified.")
        #
        # # Ensure correct types
        # if pre_cell_id is None:
        #     pass
        # elif isinstance(pre_cell_id, int):
        #     pre_cell_id = [int(pre_cell_id)]
        # else:
        #     pre_cell_id = [int(i) for i in pre_cell_id]
        # if post_cell_id is None:
        #     pass
        # elif isinstance(post_cell_id, int):
        #     post_cell_id = [int(post_cell_id)]
        # else:
        #     post_cell_id = [int(i) for i in post_cell_id]
        #
        # # RETURN SYNAPSES BETWEEN NEURONS
        # if pre_cell_id is not None and post_cell_id is not None:
        #     # df = fetch_synapse_connections(source_criteria=NC(bodyId=pre_cell_id),
        #     #                                target_criteria=NC(bodyId=post_cell_id),
        #     #                                synapse_criteria=SC(rois=neuropil))
        #     df = self._fetch_synapse_connections(pre_cell_id, post_cell_id, neuropil)
        #     return df[["bodyId_pre", "bodyId_post", "roi_post", "x_post", "y_post", "z_post"]]
        #
        # # RETURN INPUT SYNAPSES
        # if pre_cell_id is None and post_cell_id is not None:
        #     df = self._fetch_synapses(post_cell_id, neuropil, "post")
        #     df["dummy"] = np.nan
        #     return df[["dummy", "bodyId", "roi", "x", "y", "z"]]
        # # RETURN OUTPUT SYNAPSES
        # if pre_cell_id is not None and post_cell_id is None:
        #     df = self._fetch_synapses(pre_cell_id, neuropil, "pre")
        #     df["dummy"] = np.nan
        #     return df[["bodyId", "dummy", "roi", "x", "y", "z"]]
