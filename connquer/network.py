from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spmatrix
import pandas as pd

from typing import Union


def indices_of_values(values: np.ndarray, options: np.ndarray) -> (np.ndarray, np.ndarray):
    """Computes the indices (i, j) required to order two 1d arrays such that values[i] == options[j]."""
    try:
        assert (np.unique(options).size == options.size)  # checks that all options are unique
    except TypeError:
        pass
    val_to_idx = {v: i for (i, v) in enumerate(options)}
    valid = np.isin(values, options)
    i, = np.where(valid)
    j = np.array([val_to_idx[values[idx]] for idx in i])
    return i.astype("i4"), j.astype("i4")


class LabeledConnections:
    """Class for sparse matrices with labeled rows and columns.

    Parameters
    ----------
    coo : coo_matrix
        A scipy sparse matrix (coo, csc or csr format).
    rows: np.ndarray
        A 1D numpy array with row labels for the coo.
    cols: np.ndarray
        A 1D numpy array with column labels for the coo.
    """

    def __init__(self, coo: spmatrix, rows: np.ndarray, cols: np.ndarray):
        assert coo.shape == (len(rows), len(cols)), "row and column labels must match matrix shape."
        self._coo = coo_matrix(coo)
        self.rows = np.array(rows)
        self.cols = np.array(cols)

    def __repr__(self):
        return f"{type(self).__name__}({self.coo.shape})"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, row_name="pre_cell_id", col_name="post_cell_id", weight_name="syn_count") -> LabeledConnections:
        """Construct a labeled connections object from dataframe."""
        row_labels = df[row_name].unique()
        col_labels = df[col_name].unique()
        _, idxs = indices_of_values(df[row_name].values, row_labels)
        _, jdxs = indices_of_values(df[col_name].values, col_labels)
        weights = df[weight_name].values
        conn = coo_matrix((weights, (idxs, jdxs)), shape=(len(row_labels), len(col_labels)))
        conn.sum_duplicates()
        return cls(conn, row_labels, col_labels)

    @classmethod
    def from_dense(cls, a: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> LabeledConnections:
        """Construct a labeled connections object from a dense array."""
        i, j = np.where(a)
        vals = a[i, j]
        return cls(coo_matrix((vals, (i, j)), shape=a.shape), rows, cols)

    @classmethod
    def null(cls, dtype="int32", rows=None, cols=None):
        """Construct a labeled connections object with null values."""
        # if rows is None and cols is None:
        #     return cls(coo_matrix((0, 0)), np.array([], dtype=dtype), np.array([], dtype=dtype))
        if rows is None:
            rows = np.array([])
        if cols is None:
            cols = np.array([])
        return cls(coo_matrix((len(rows), len(cols)), dtype=dtype), rows, cols)

    @property
    def shape(self) -> tuple[int, int]:
        return self.coo.shape

    @property
    def rows(self) -> np.ndarray:
        return self._rows

    @rows.setter
    def rows(self, rows):
        assert len(rows) == self.shape[0]
        self._rows = rows
        self._rowmap = {l: i for i, l in enumerate(rows)}

    @property
    def cols(self) -> np.ndarray:
        return self._cols

    @cols.setter
    def cols(self, cols):
        assert len(cols) == self.shape[1]
        self._cols = cols
        self._colmap = {l: j for j, l in enumerate(cols)}

    @property
    def coo(self) -> coo_matrix:
        return self._coo

    @property
    def csr(self) -> csr_matrix:
        return self.coo.tocsr()

    @property
    def csc(self) -> csc_matrix:
        return self.coo.tocsc()

    @property
    def T(self) -> LabeledConnections:
        return LabeledConnections(self.coo.T, self.cols, self.rows)

    def row_indices(self, vals) -> np.ndarray:
        return np.array([self._rowmap[val] for val in np.atleast_1d(vals)])

    def col_indices(self, vals) -> np.ndarray:
        return np.array([self._colmap[val] for val in np.atleast_1d(vals)])

    def row_intersect(self, vals):
        return np.array([val for val in vals if val in self.rows])

    def col_intersect(self, vals):
        return np.array([val for val in vals if val in self.cols])

    def take(self, rows=None, cols=None):
        """Like __getitem__, but ignores rows and columns not in the underlying data."""
        rows = self.row_intersect(rows) if rows is not None else slice(None, None, None)
        cols = self.col_intersect(cols) if cols is not None else slice(None, None, None)
        return self[rows, cols]

    def get_data(self, as_dense=True) -> (Union[coo_matrix, np.ndarray], np.ndarray, np.ndarray):
        """Return underlying labeled data."""
        a = self.coo.toarray() if as_dense else self.coo
        return a, self.rows, self.cols

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, cols = item
        else:
            rows = item
            cols = slice(None, None, None)
        if not isinstance(rows, slice):
            rows = np.array([self._rowmap[i] for i in np.atleast_1d(rows)], dtype="int32")
        if not isinstance(cols, slice):
            cols = np.array([self._colmap[i] for i in np.atleast_1d(cols)], dtype="int32")
        return LabeledConnections(self.csr[rows].tocsc()[:, cols], self.rows[rows], self.cols[cols])

    def __matmul__(self, other) -> LabeledConnections:
        if not isinstance(other, LabeledConnections):
            raise TypeError(f"Expected LabeledConnections, got {type(other)}")
        left_cols = self.cols
        right_rows = other.rows
        try:
            shared, j, i = np.intersect1d(left_cols, right_rows, assume_unique=True, return_indices=True)
        except TypeError:
            j, i = indices_of_values(left_cols, right_rows)
        new = self.csc[:, j].tocsr() @ other.csr[i].tocsc()
        return type(self)(new, self.rows, other.cols)

    def normalize(self, axis=0) -> LabeledConnections:
        """Normalize values other either rows (which='rows') or columns (which='cols')."""
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1.")
        mat = self.csr if axis else self.csc
        a_normed = mat / mat.sum(axis=axis)
        return LabeledConnections(a_normed, self.rows, self.cols)

    def sum(self, axis=0, return_labels=False) -> Union[np.matrix, (np.matrix, np.ndarray)]:
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1.")
        mat = self.csr if axis else self.csc
        s = mat.sum(axis=axis)
        if return_labels:
            labels = self.rows if axis else self.cols
            return s, labels
        return s

    def __add__(self, other: LabeledConnections):
        if not isinstance(other, LabeledConnections):
            raise TypeError(f"Expected LabeledConnections, got {type(other)}")
        try:
            rows = np.union1d(self.rows, other.rows)
            cols = np.union1d(self.cols, other.cols)
        except TypeError:
            rows = np.array(list(set(self.rows).union(set(other.rows))))
            cols = np.array(list(set(self.cols).union(set(other.cols))))
        if not other:
            return self
        if not self:
            return other
        # Left indices and values
        _, i_left = indices_of_values(self.rows[self.coo.row], rows)
        _, j_left = indices_of_values(self.cols[self.coo.col], cols)
        v_left = self.coo.data
        # Right indices and values
        _, i_right = indices_of_values(other.rows[other.coo.row], rows)
        _, j_right = indices_of_values(other.cols[other.coo.col], cols)
        v_right = other.coo.data
        # New matrix
        i = np.concatenate((i_left, i_right))
        j = np.concatenate((j_left, j_right))
        v = np.concatenate((v_left, v_right))
        coo = coo_matrix((v, (i, j)), shape=(len(rows), len(cols)))
        coo.sum_duplicates()
        return type(self)(coo, rows, cols)

    def drop_empty(self, rows=False, cols=False):
        """Removes empty rows and/or columns from the matrix."""
        a = self
        if rows:
            total, rows = a.sum(axis=1, return_labels=True)
            nonzero, _ = np.nonzero(total.A)
            a = a[rows[nonzero]]  #type(self)(a.csr[nonzero], self.row_labels[nonzero], self.col_labels)
        if cols:
            total, cols = a.sum(axis=0, return_labels=True)
            _, nonzero = np.nonzero(total.A)
            a = a[:, cols[nonzero]]
        return a

    @staticmethod
    def _combiner(row_dict) -> LabeledConnections:
        rows = []
        cols = []
        vals = []
        for k, v in row_dict.items():
            rows.extend([k] * len(v))
            cols.extend(v)
            vals.extend(np.ones(len(v)))
        df = pd.DataFrame(list(zip(rows, cols, vals)), columns=["row", "col", "val"])
        conn = LabeledConnections.from_dataframe(df, "row", "col", "val")
        return conn

    def combine(self, method="sum", row_dict: dict=None, col_dict: dict=None):
        if method not in ("sum", "mean"):
            raise ValueError(f"Method must be 'sum' or 'mean', got {method}")
        conn = self
        if row_dict is not None:
            left = self._combiner(row_dict)
            if method == "mean":
                left = left.normalize(axis=1)
            conn = left @ conn
        if col_dict is not None:
            right = self._combiner(col_dict).T
            if method == "mean":
                right = right.normalize(axis=0)
            conn = conn @ right
        return conn

    def __bool__(self):
        return self.shape != (0, 0)


if __name__ == "__main__":
    n = m = 10000
    k = 1000
    i = np.random.randint(0, n, k)
    j = np.random.randint(0, m, k)
    a1 = LabeledConnections(coo_matrix((np.ones(k), (i, j)), (n, m)), np.arange(n), np.arange(m))
    a1 = a1.drop_empty(rows=True, cols=True)
    a2 = LabeledConnections(coo_matrix((np.ones(k), (i, j)), (n, m)), np.arange(n, 2 * n), np.arange(m))
    a2 = a2.drop_empty(rows=True, cols=True)
    b = a1 + a2
    print(b.sum(axis=0, return_labels=True))
