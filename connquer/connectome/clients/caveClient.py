from ..dataset import Connectome, ClientMixin, NameSpaceMap
import caveclient
import pandas as pd
from typing import Union, Iterable
from numpy import ndarray


class CaveClient(Connectome, ClientMixin):
    """Class for accessing online datasets through cave."""

    _synapse_table = 'valid_synapses_nt_np_v6'

    NAMESPACE = NameSpaceMap()

    @property
    def client(self):
        return self._c

    def connect_to_client(self, token, dataset):
        return caveclient.CAVEclient(dataset, auth_token=token)

    def load_brain_regions(self) -> list:
        raise NotImplementedError

    def load_cell_types(self) -> dict:
        raise NotImplementedError

    def _get_cell_ids(self, cell_type: str) -> list:
        raise NotImplementedError

    def _get_id_types(self, cell_ids: Union[int, Iterable[int]]) -> pd.DataFrame:
        raise NotImplementedError

    def _get_neurotransmitter(self, cell_type):
        return NotImplemented

    def _get_connections(self, *, pre_ids: Union[int, Iterable[int]] = None, post_ids: Union[int, Iterable[int]] = None,
                         regions: Union[str, Iterable[str]] = None, threshold: int = None) -> pd.DataFrame:
        raise NotImplementedError

    def _get_synapses(self, *,
                     pre_cell_id: Union[int, Iterable, ndarray] = None,
                     post_cell_id: Union[int, Iterable, ndarray] = None,
                     neuropil: Union[str, Iterable] = None) -> pd.DataFrame:
        filter_dict = {}
        # FILTER BY PRESYNAPTIC CELL
        if pre_cell_id is None:
            pass
        elif isinstance(pre_cell_id, int):
            filter_dict["pre_pt_root_id"] = [pre_cell_id]
        else:
            filter_dict["pre_pt_root_id"] = pre_cell_id
        # FILTER BY POSTSYNAPTIC CELL
        if post_cell_id is None:
            pass
        elif isinstance(post_cell_id, int):
            filter_dict["post_pt_root_id"] = [post_cell_id]
        else:
            filter_dict["post_pt_root_id"] = post_cell_id
        # FILTER BY NEUROPIL
        if neuropil is None:
            pass
        elif isinstance(neuropil, str):
            filter_dict["neuropil"] = [neuropil]
        else:
            filter_dict["neuropil"] = neuropil
        # QUERY CAVE CLIENT
        df = self.client.materialize.query_view(self._synapse_table, filter_in_dict=filter_dict)
        pos = pd.DataFrame(df["post_pt_position"].tolist(), columns=["x", "y", "z"], index=df.index)
        df = pd.concat([df, pos], axis=1)
        return df[["pre_pt_root_id", "post_pt_root_id", "neuropil", "x", "y", "z"]]
