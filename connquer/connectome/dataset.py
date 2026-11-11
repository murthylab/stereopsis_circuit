from abc import ABC, abstractmethod
import pandas as pd
from typing import Union, Iterable
from numpy import ndarray
from dataclasses import dataclass
from functools import wraps


class Cached:
    """Descriptor for caching large data files."""

    def __init__(self, getter, args=(), kwargs=None):
        self.getter = getter
        self.args = args
        self.kwargs = kwargs or {}

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, instance, owner):
        if hasattr(instance, self.private_name):
            return getattr(instance, self.private_name)
        value = getattr(instance, self.getter)(*self.args, **self.kwargs)
        setattr(instance, self.private_name, value)
        return value


@dataclass
class NameSpaceMap:
    """Class for mapping different datasets to a common namespace."""

    cell_type: str = "cell_type"
    cell_id: str = "cell_id",
    pre_cell_id: str = "pre_cell_id"
    post_cell_id: str  = "post_cell_id"
    side: str = "side"
    neuropil: str = "neuropil"
    syn_count: str  = "syn_count"
    x: str = "x"
    y: str = "y"
    z: str = "z"

    def map(self):
        return {val: k for (k, val) in self.__dict__.items()}

    def inverse_map(self):
        return self.__dict__


class ClientMixin(ABC):
    """Mixin class for accessing online datasets."""

    def __init__(self, token=None, dataset=None):
        self._c = self.connect_to_client(token, dataset)

    @abstractmethod
    def connect_to_client(self, token, dataset):
        pass


def common_namespace(method):
    """Decorator function that maps DataFrames to a common namespace."""
    @wraps(method)
    def wrapper(obj, *args, **kwargs):
        df = method(obj, *args, **kwargs)
        return df.rename(columns=obj.NAMESPACE.map())
    return wrapper


class Connectome(ABC):
    """Base class for interacting with connectomic datasets."""

    cell_types = Cached("load_cell_types")
    brain_regions = Cached("load_brain_regions")

    NAMESPACE: NameSpaceMap = NotImplemented

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.NAMESPACE is NotImplemented:
            raise TypeError(
                f"{cls.__name__} must define NAMESPACE"
            )

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"

    def get_cell_ids(self, cell_type: str) -> ndarray:
        """Returns a list of all cell ids of the corresponding type.

        Parameters
        ----------
        cell_type : str
            The cell type to get cell ids for.

        Returns
        -------
        cell_ids : ndarray
        """
        return self._get_cell_ids(cell_type)

    def get_neurotransmitter(self, cell_type):
        """Returns the consolidated neurotransmitter of a given cell type.

        Parameters
        ----------
        cell_type : str
            The cell type to get the consolidated predicted neurotransmitter identity.

        Returns
        -------
        neurotransmitter : str
        """
        return self._get_neurotransmitter(cell_type)

    @common_namespace
    def get_id_types(self, cell_ids: Union[int, Iterable[int]]) -> pd.DataFrame:
        """Returns a dataframe cell types and side of the corresponding cell_ids.

        Parameters
        ----------
        cell_ids : Union[int, Iterable[int]]
            A sequence of cell ids to get cell types for.

        Returns
        -------
        id_types : pd.DataFrame
            A dataframe of cell types and soma side of the corresponding cell_ids.
        """
        return self._get_id_types(cell_ids)

    @common_namespace
    def get_synapses(self, *, pre_ids: Union[int, Iterable[int], ndarray] = None,
                      post_ids: Union[int, Iterable[int], ndarray] = None,
                      **kwargs) -> pd.DataFrame:
        """Returns a dataframe of synapse locations.

        Parameters
        ----------
        pre_ids : Union[int, Iterable[int], ndarray]
            Pre-synaptic (upstream) cell ids.
        post_ids : Union[int, Iterable[int], ndarray]
            Post-synaptic (downstream) cell ids.
        """
        return self._get_synapses(pre_ids=pre_ids, post_ids=post_ids, **kwargs)

    @common_namespace
    def get_connections(self, *, pre_ids: Union[int, Iterable[int]]=None, post_ids: Union[int, Iterable[int]]=None,
                        regions: Union[str, Iterable[str]]=None, threshold: int=None):
        """Returns a dataframe of connections.

        Parameters
        ----------
        pre_ids : Union[int, Iterable[int], ndarray]
            Pre-synaptic (upstream) cell ids.
        post_ids : Union[int, Iterable[int], ndarray]
            Post-synaptic (downstream) cell ids.
        regions : Union[str, Iterable[str]]
            Restrict connections to the given brain regions / neuropils.
        threshold : int
            Synapse threshold to apply to connection search.

        Returns
        -------
        connections : pd.DataFrame
            A dataframe of all connection weights (syn_count) between the given pre- and post-cell ids in the given
            brain regions at the given synapse threshold.
        """
        df = self._get_connections(pre_ids=pre_ids, post_ids=post_ids, regions=regions, threshold=threshold)
        return df.rename(columns=self.NAMESPACE.map())

    def get_inputs(self, cell_ids: Union[int, Iterable[int]], regions: Union[str, Iterable[str]]=None, threshold: int = None) -> pd.DataFrame:
        """Returns a dataframe of presynaptic partners with number of synapses per neuropil."""
        return self.get_connections(post_ids=cell_ids, regions=regions, threshold=threshold)

    def get_outputs(self, cell_ids: Union[int, Iterable[int]], regions: Union[str, Iterable[str]]=None, threshold: int = None) -> pd.DataFrame:
        """Returns a dataframe of postsynaptic partners with number of synapses per neuropil."""
        return self.get_connections(pre_ids=cell_ids, regions=regions, threshold=threshold)

    @abstractmethod
    def load_brain_regions(self) -> list:
        """Returns a list brain regions in the dataset."""
        pass

    @abstractmethod
    def load_cell_types(self) -> dict:
        """Returns a dictionary of all cell types and the corresponding cell_ids."""
        pass

    @abstractmethod
    def _get_neurotransmitter(self, cell_type):
        pass

    @abstractmethod
    def _get_cell_ids(self, cell_type: str) -> ndarray:
        pass

    @abstractmethod
    def _get_id_types(self, cell_ids: Union[int, Iterable[int]]) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_connections(self, *, pre_ids: Union[int, Iterable[int]] = None, post_ids: Union[int, Iterable[int]] = None,
                         regions: Union[str, Iterable[str]] = None, threshold: int = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def _get_synapses(self, *, pre_cell_ids: Union[int, Iterable, ndarray] = None,
                      post_cell_ids: Union[int, Iterable, ndarray] = None,
                      **kwargs) -> pd.DataFrame:
        pass
