from .model import EyeModel
import numpy as np
from sklearn.kernel_ridge import KernelRidge


class DatasetMapper:

    def __init__(self, eye_model, poly_degree=3):
        self.eye_model = eye_model
        self.poly_degree = poly_degree
        self.centroid_ = None
        self.sigmas_ = None
        self.components_ = None
        self.kr_model_ = None

    def fit(self):
        # Compute principal components
        self.centroid_ = self.eye_model.pq.mean(axis=0)
        u, self.sigmas_, vt = np.linalg.svd(self.eye_model.pq - self.centroid_)
        sign = np.sign(np.diag(vt))
        self.components_ = vt * sign[:, None]
        # KernelRidge regression
        lu = np.hstack([self.eye_model.xyz_l, self.eye_model.xyz_u])
        self.kr_model_ = KernelRidge(kernel="polynomial", degree=self.poly_degree)
        self.kr_model_.fit(self.eye_model.pq, lu)
        return self

    def transform(self, pq, renormalize=True, return_object=True):
        if self.kr_model_ is None:
            raise ValueError("No model fitted")
        if renormalize:
            x, s, vt = np.linalg.svd(pq - pq.mean(axis=0))
            sign = np.sign(np.diag(vt))
            x = x[:, :2] * sign[None]
            pq_mapped = (x @ np.diag(self.sigmas_) @ self.components_) + self.centroid_
        else:
            pq_mapped = pq
        lu = self.kr_model_.predict(pq_mapped)
        xyz_l = lu[:, :3]
        xyz_u = lu[:, 3:]
        xyz_u = xyz_u / np.linalg.norm(xyz_u, axis=-1, keepdims=True)
        if not return_object:
            return dict(xyz_l=xyz_l, xyz_u=xyz_u, pq=pq, pq_mapped=pq_mapped)
        xyz_c = xyz_l - xyz_u
        params = self.eye_model.get_params()
        params.update(xyz_l=xyz_l, xyz_c=xyz_c, pq=pq)
        return EyeModel(**params)

    def fit_transform(self, pq, **kwargs):
        self.fit()
        return self.transform(pq, **kwargs)
