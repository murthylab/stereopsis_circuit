from connquer.network import LabeledConnections
import numpy as np
from scipy.sparse import coo_matrix


class VisualColumnConnections(LabeledConnections):

    @classmethod
    def from_column_ids(cls, column_ids, cell_ids):
        assert len(column_ids) == len(cell_ids)
        return cls(cls.eye(len(column_ids)), column_ids, cell_ids)

    @staticmethod
    def eye(n):
        i = j = np.arange(n)
        return coo_matrix((np.ones(n), (i, j)), shape=(n, n))


class VisualColumnMap:

    def __init__(self, df):
        self.df = df
        self._id_to_pq, self._pq_to_id = self._get_id_pq_transforms(self.df)

    @property
    def column_ids(self):
        return self.df["column_id"].unique()  # note: column ids start at 1

    @property
    def cell_types(self):
        return self.df["cell_type"].unique()

    @property
    def cell_ids(self):
        return self.df["cell_id"].unique()

    def pq_to_id(self, pq):
        pq = np.atleast_2d(pq)
        return self._pq_to_id[pq[:, 0], pq[:, 1]]

    def id_to_pq(self, col_ids):
        return np.array([self._id_to_pq[i] for i in np.atleast_1d(col_ids)])

    @staticmethod
    def _get_id_pq_transforms(df) -> (dict, np.ndarray):
        n = len(df["p"].unique())
        m = len(df["q"].unique())
        id_to_pq = {}
        pq_to_id = np.zeros((n, m))
        for col_id, pq in df.groupby("column_id")[["p", "q"]]:
            p, q = pq.values[0]
            id_to_pq[col_id] = (p, q)
            pq_to_id[p, q] = col_id
        return id_to_pq, pq_to_id

    def __getitem__(self, item):
        if item == slice(None):
            conn = LabeledConnections.null()
            for cell_type in self.cell_types:
                conn = conn + self[cell_type]
            return conn
        assert isinstance(item, str)
        if item not in self.cell_types:
            raise KeyError(f"'{item}' is not a columnar type")
        df = self.df.query("cell_type == @item")
        column_ids = df["column_id"].values
        cell_ids = df["cell_id"].values
        return VisualColumnConnections.from_column_ids(column_ids, cell_ids)
