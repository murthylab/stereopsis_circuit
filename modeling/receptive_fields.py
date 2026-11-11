from parameters import DATASETS, hexmap
import h5py
import numpy as np
import pandas as pd
from eye_model_3d import MappedDataset


def load_receptive_fields(directory, left_column_ids, right_column_ids, include_contra=False):
    # ===============
    # Load DataFrames
    # ===============

    rfs_left = pd.read_csv(directory / "exc_left_ipsi_full.csv", index_col=0)
    rfs_left_i = pd.read_csv(directory / "inh_left_ipsi_full.csv", index_col=0)

    rfs_right = pd.read_csv(directory / "exc_right_ipsi_full.csv", index_col=0)
    rfs_right_i = pd.read_csv(directory / "inh_right_ipsi_full.csv", index_col=0)

    rf_stats = pd.read_csv(directory / "receptive_field_properties.csv")

    # ====================
    # Take connected pairs
    # ====================
    # if include_contra:
    #     id_pairs = []
    #     for n in range(1, 16):
    #         left_name = f"LC14a1-L{n:02}"
    #         right_name = f"LC14a1-R{n:02}"
    #         left_id = rf_stats.query("name == @left_name").iloc[0].cell_id
    #         right_id = rf_stats.query("name == @right_name").iloc[0].cell_id
    #         id_pairs.append((str(left_id), str(right_id)))
    #     left_order, right_order = zip(*id_pairs)
    # else:
    left_order = rf_stats[[x.split("-")[1].startswith("L") for x in rf_stats["name"].values]].cell_id.values
    left_order = [str(i) for i in left_order if str(i) in rfs_left_i.columns]
    right_order = rf_stats[[x.split("-")[1].startswith("R") for x in rf_stats["name"].values]].cell_id.values
    right_order = [str(i) for i in right_order if str(i) in rfs_right_i.columns]

    rfs_left = rfs_left[list(left_order)]
    rfs_right = rfs_right[list(right_order)]

    rfs_left_i = rfs_left_i[list(left_order)]
    rfs_right_i = rfs_right_i[list(right_order)]

    # ===============
    # Reorder neurons
    # ===============

    rf_weights = {}
    rf_weights["left_exc"] = rfs_left.loc[left_column_ids].values
    rf_weights["left_inh"] = rfs_left_i.loc[left_column_ids].values
    rf_weights["right_exc"] = rfs_right.loc[right_column_ids].values
    rf_weights["right_inh"] = rfs_right_i.loc[right_column_ids].values

    if include_contra:

        rfs_left_c_df = pd.read_csv(directory / "inh_left_contra_full.csv", index_col=0)
        rfs_right_c_df = pd.read_csv(directory / "inh_right_contra_full.csv", index_col=0)

        rfs_left_c = pd.DataFrame(data=0, index=rfs_right.index, columns=rfs_left.columns).add(rfs_left_c_df,
                                                                                               fill_value=0)
        rfs_right_c = pd.DataFrame(data=0, index=rfs_left.index, columns=rfs_right.columns).add(rfs_right_c_df,
                                                                                                fill_value=0)

        rfs_left_c = rfs_left_c[list(left_order)]
        rfs_right_c = rfs_right_c[list(right_order)]

        rf_weights["left_con"] = rfs_left_c.loc[right_column_ids].values
        rf_weights["right_con"] = rfs_right_c.loc[left_column_ids].values

    return rf_weights


def gaussian_approximation_truncated(xy, weights):
    """Estimate the underlying 2D Gaussian from a half-space-truncated Gaussian.

    Assumes:
        - truncation boundary passes through the true Gaussian mean
        - points satisfy normal @ (x - mean) >= 0
        - `normal` is the normal vector to the truncation boundary
    """
    xy = np.atleast_2d(xy)
    weights = np.atleast_1d(weights)

    assert xy.shape[0] == weights.shape[-1]

    # Normalize weights
    weights_norm = weights / weights.sum(axis=-1, keepdims=True)
    weights_norm = np.nan_to_num(weights_norm)

    # Observed (truncated) centroid
    centroid_obs = np.einsum(
        "ni,...n->...ni",
        xy,
        weights_norm,
        optimize="greedy"
    ).sum(axis=-2)

    # Center points around observed centroid
    xy_centered = xy - centroid_obs[..., None, :]

    # Observed covariance
    cov_obs = np.einsum(
        "...n,...ni,...nj->...ij",
        weights_norm,
        xy_centered,
        xy_centered
    ) + (np.eye(2) * (5. / 36.))

    normal = -centroid_obs / np.linalg.norm(centroid_obs)

    # Normalize truncation direction
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # C @ n
    cov_n = np.einsum("...ij,...j->...i", cov_obs, normal)
    n_cov_n = np.einsum("...i,...i->...", normal, cov_n)

    # Recover underlying mean
    centroid = centroid_obs - np.sqrt(2 / np.pi) * cov_n / np.sqrt(n_cov_n[:, None])

    # Variance in normal direction for the underlying Gaussian
    cov_nn_cov = np.einsum("...ij,...j,...k,...kl->...il", cov_obs, normal, normal, cov_obs)

    # Recover underlying covariance
    cov = cov_obs + (2 / np.pi) * cov_nn_cov / n_cov_n[:, None, None]

    return centroid, cov


def generate_weights(xy, centroid, cov):
    xy_centered = xy - centroid[..., None, :]
    q_matrix = np.linalg.inv(cov)
    q_matrix[np.isnan(q_matrix)] = 0
    gauss_weight = np.exp(-0.5 * np.einsum("...ni,...ij,...nj->...n", xy_centered, q_matrix, xy_centered))
    return gauss_weight


if __name__ == "__main__":
    # Load dataset
    dset = DATASETS["flywire"]
    directory = dset.directory / "modeling"
    directory.mkdir(parents=True, exist_ok=True)

    # Load eye model
    eye_model = MappedDataset.load("flywire", convergence=np.radians(5))
    xy_l = hexmap(*eye_model["left"].pq.T)
    xy_r = hexmap(*eye_model["right"].pq.T)

    # Load receptive fields
    rf_raw = load_receptive_fields(dset.directory / "receptive_fields" / "LC14a1", eye_model["left"].column_ids,
                                   eye_model["right"].column_ids, include_contra=True)

    # Compute smooth receptive field
    smooth_rfs = {}
    for rf_type in ("exc", "inh"):
        centroids_l, covs_l = gaussian_approximation_truncated(xy_l, rf_raw[f"left_{rf_type}"].T)
        centroids_r, covs_r = gaussian_approximation_truncated(xy_r, rf_raw[f"right_{rf_type}"].T)
        centroids = np.concatenate((centroids_l, centroids_r), axis=0)
        covs = np.concatenate((covs_l, covs_r), axis=0)
        det = np.linalg.det(covs)
        det_mean = np.mean(det)
        scale = np.sqrt(det / det_mean)
        covs = covs / scale[:, None, None]

        rf_l_smooth = generate_weights(xy_l, centroids[:15], covs[:15])
        rf_r_smooth = generate_weights(xy_r, centroids[15:], covs[15:])

        rf_l_smooth = rf_l_smooth / (2 * np.pi * np.sqrt(det_mean))
        rf_r_smooth = rf_r_smooth / (2 * np.pi * np.sqrt(det_mean))

        smooth_rfs[rf_type] = (rf_l_smooth, rf_r_smooth)

    # Save
    with h5py.File(directory / "receptive_fields.h5", "w") as f:
        f.create_dataset("left_exc", data=smooth_rfs["exc"][0])
        f.create_dataset("right_exc", data=smooth_rfs["exc"][1])
        f.create_dataset("left_inh", data=smooth_rfs["inh"][0])
        f.create_dataset("right_inh", data=smooth_rfs["inh"][1])
