from parameters import DATASETS, contralateral
from connquer.network import LabeledConnections
import pandas as pd
import h5py


def compute_feedback(dset):
    """Compute recurrent feedback within the LC14a1 network. Returns a dictionary of pairwise coupling matrices between
    LC14a1 neurons."""

    rf_stats = pd.read_csv(dset.directory / "receptive_fields" / "LC14a1" / "receptive_field_properties.csv")

    K = {}  # coupling matrices

    for side in ("left", "right"):

        # Get LC14a1 cell ids
        cell_ids = rf_stats.query("side == @side")["cell_id"].values
        contra = contralateral[side]  # contralateral side
        contra_ids = rf_stats.query("side == @contra")["cell_id"].values
        neuropils = dset.neuropil_formatter("LO", side)
        neuropils_contra = dset.neuropil_formatter("LO", contra)
        # neuropils = [dset.neuropil_formatter(s, side) for s in dset.optic_lobe_neuropils]
        # neuropils_contra = [dset.neuropil_formatter(s, contra) for s in dset.optic_lobe_neuropils]

        # Get monosynaptic input to LC14a1
        monosynaptic = dset.data.get_inputs(cell_ids, regions=neuropils, threshold=dset.THRESHOLD)
        monosynaptic = LabeledConnections.from_dataframe(monosynaptic).normalize(axis=0)  # normalize
        # Take excitatory inputs
        ct_df = dset.data.get_id_types(monosynaptic.rows)
        ct_df = ct_df[~pd.isnull(ct_df["cell_type"])]
        exc_cts = [ct for ct in ct_df["cell_type"].unique() if dset.data.get_neurotransmitter(ct) == "ACH"]
        exc_ids = ct_df.query("cell_type in @exc_cts")["cell_id"].values
        monosynaptic_exc = monosynaptic[exc_ids]
        # Take inhibitory inputs
        inh_cts = [ct for ct in ct_df["cell_type"].unique() if dset.data.get_neurotransmitter(ct) in ("GLUT", "GABA")]
        inh_ids = ct_df.query("cell_type in @inh_cts")["cell_id"].values
        monosynaptic_inh = monosynaptic[inh_ids]
        # Get contralateral inputs
        monosynaptic_contra = dset.data.get_inputs(cell_ids, regions=neuropils_contra, threshold=dset.THRESHOLD)
        monosynaptic_contra = LabeledConnections.from_dataframe(monosynaptic_contra).normalize(axis=0)

        # Compute disynaptic excitatory input
        disynaptic_exc = dset.data.get_inputs(monosynaptic_exc.rows, regions=neuropils, threshold=dset.THRESHOLD)
        disynaptic_exc = LabeledConnections.from_dataframe(disynaptic_exc).normalize(axis=0)
        ct_df = dset.data.get_id_types(disynaptic_exc.rows)
        ct_df = ct_df[~pd.isnull(ct_df["cell_type"])]
        exc_cts = [ct for ct in ct_df["cell_type"].unique() if dset.data.get_neurotransmitter(ct) == "ACH"]
        exc_ids = ct_df.query("cell_type in @exc_cts")["cell_id"].values
        disynaptic_exc = disynaptic_exc[exc_ids]

        # Compute trisynaptic excitatory input
        trisynaptic_exc = dset.data.get_inputs(disynaptic_exc.rows, regions=neuropils, threshold=dset.THRESHOLD)
        trisynaptic_exc = LabeledConnections.from_dataframe(trisynaptic_exc).normalize(axis=0)

        # Compute total disynaptic and trisynaptic excitatory input
        di_coupling = disynaptic_exc @ monosynaptic_exc
        tri_coupling = trisynaptic_exc @ di_coupling
        # Sum monosynaptic, disynaptic and trisynaptic input from contralateral LC14a1
        mono = monosynaptic_exc.take(rows=contra_ids, cols=cell_ids)
        di = di_coupling.take(rows=contra_ids, cols=cell_ids)
        tri = tri_coupling.take(rows=contra_ids, cols=cell_ids)
        total_coupling = (mono + di + tri)[contra_ids, cell_ids]
        # Add to result
        total_coupling, *_ = total_coupling.get_data(as_dense=True)
        K[f"VW+_{contra[0]}2{side[0]}"] = total_coupling

        # Compute disynaptic inhibitory input
        disynaptic_inh = dset.data.get_inputs(monosynaptic_inh.rows, regions=neuropils, threshold=dset.THRESHOLD)
        disynaptic_inh = LabeledConnections.from_dataframe(disynaptic_inh).normalize(axis=0)
        # Ipsilateral inhibitory input
        di_inh = (disynaptic_inh @ monosynaptic_inh).take(rows=contra_ids, cols=cell_ids)[contra_ids, cell_ids]
        di_inh, *_ = di_inh.get_data(as_dense=True)
        K[f"VW-_{contra[0]}2{side[0]}"] = di_inh

        # Compute disynaptic contralateral input
        disynaptic_contra = dset.data.get_inputs(monosynaptic_contra.rows, regions=neuropils_contra, threshold=dset.THRESHOLD)
        disynaptic_contra = LabeledConnections.from_dataframe(disynaptic_contra).normalize(axis=0)
        # Contralateral self-feedback
        self_inh = (disynaptic_contra @ monosynaptic_contra).take(rows=cell_ids, cols=cell_ids)[cell_ids, cell_ids]
        self_inh, *_ = self_inh.get_data(as_dense=True)
        K[f"VZ-_{side[0]}2{side[0]}"] = self_inh

    return K


if __name__ == "__main__":
    
    for name, dset in DATASETS.items():

        directory = dset.directory / "modeling"
        directory.mkdir(parents=True, exist_ok=True)

        feedback = compute_feedback(dset)

        with h5py.File(directory / "feedback.h5", "w") as f:
            for label, mat in feedback.items():
                f.create_dataset(label, data=mat)
