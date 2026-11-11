from .model import EyeModel
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd


__all__ = ["ZhaoDataset", "MappedDataset"]


class modelgetter:

    def __init__(self, folder):
        self.folder = folder

    def __get__(self, obj, cls):
        return {
            f.stem: f for f in Path(__file__).parent.joinpath("datasets", self.folder).iterdir()
        }


class ModelFactory(ABC):
    """Base class for loading 3D eye models."""

    model_cls = EyeModel  # class returned by factory (allows users to load instance of own subclass)
    models = modelgetter(...)  # pre-saved model files

    @classmethod
    @abstractmethod
    def from_dataframe(cls, df, **kwargs) -> EyeModel:
        ...

    @classmethod
    def load(cls, name_or_path, **kwargs) -> dict[str, EyeModel]:
        """Load an eye model from a user-defined or pre-saved file."""
        # Check if pre-saved file
        saved_models = cls.models
        if name_or_path in saved_models:
            path = saved_models[name_or_path]
        else:
            path = Path(name_or_path)
            if not path.exists():
                raise FileNotFoundError(f"Path {name_or_path!r} not found")
        # Load dataframe
        df = pd.read_csv(path)
        # Generate dictionary of models for left and right eyes
        model = {}
        for side, df_side in df.groupby("side"):
            kw = dict(kwargs)
            if side == "right":
                kw["is_right"] = True
            model[side] = cls.from_dataframe(df_side, **kw)
        return model


class ZhaoDataset(ModelFactory):

    models = modelgetter("zhao_et_al_2025")

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        xyz_l = df[["lens_x", "lens_y", "lens_z"]].values
        xyz_c = df[["cone_x", "cone_y", "cone_z"]].values
        pq = df[["p", "q"]].values.astype("i4")
        return cls.model_cls(xyz_l, xyz_c, pq, **kwargs)


class MappedDataset(ModelFactory):

    models = modelgetter("mapped")

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        xyz_l = df[["x", "y", "z"]].values
        xyz_u = df[["dx", "dy", "dz"]].values
        xyz_c = xyz_l - xyz_u
        pq = df[["p", "q"]].values.astype("i4")
        columns = df["column_id"].values
        return cls.model_cls(xyz_l, xyz_c, pq, columns, **kwargs)
