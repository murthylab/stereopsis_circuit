from .network import LabeledConnections
from .connectome import Connectome

class Query:

    contralateral = dict(left="right", right="left")

    def __init__(self, dataset: Connectome, threshold: int):
        self.dataset = dataset
        self.threshold = threshold

    def inputs_by_id(self, ids, regions=None):
        if not len(ids):
            raise ValueError("ids must be specified")
        # Get inputs
        input_df = self.dataset.get_connections(post_ids=ids, regions=regions, threshold=self.threshold)
        inputs_by_id = LabeledConnections.from_dataframe(input_df)
        inputs_by_id = inputs_by_id.normalize(axis=0)
        return inputs_by_id

    def group_rows_by_cell_type(self, conn: LabeledConnections, side=None):
        if side not in ("left", "right", None):
            raise ValueError("side must be 'left', 'right' or None")
        input_cell_types = self.dataset.get_id_types(conn.rows)
        cell_type_dict = {}
        if side is None:
            for cell_type, df in input_cell_types.groupby("cell_type"):
                cell_type_dict[cell_type] = df["cell_id"].values
        else:
            for (cell_type, input_side), df in input_cell_types.groupby(["cell_type", "side"]):
                if input_side == side:
                    cell_type_dict[cell_type] = df["cell_id"].values
                else:
                    cell_type_dict[cell_type + "_contra"] = df["cell_id"].values
        inputs_by_type = conn.combine(method="sum", row_dict=cell_type_dict)
        return inputs_by_type

    def cell_type_inputs(self, cell_type, regions: dict = None):
        regions = regions or {}

        # Get left and right cell ids
        query_ids = self.dataset.get_cell_ids(cell_type)
        cell_type_info = self.dataset.get_id_types(query_ids)

        inputs_dict = {}
        for side in ("left", "right"):
            inputs_dict[side] = {}
            ids = cell_type_info.query("side == @side")["cell_id"].values
            if not len(regions):
                inputs_by_id = self.inputs_by_id(ids)
                inputs_dict[side] = self.group_rows_by_cell_type(inputs_by_id, side)
                continue
            for k, neuropils in regions.items():
                inputs_by_id = self.inputs_by_id(ids, regions=neuropils)
                if inputs_by_id:
                    inputs_dict[side][k] = self.group_rows_by_cell_type(inputs_by_id, side)

        return inputs_dict
