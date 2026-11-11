import numpy as np
from pathlib import Path
import json
import re

from connquer.datasets import Flywire, Neuprint
from neuprint import fetch_neurons, NeuronCriteria as NC


def reformat_type_string(s):
    m = re.match(r'^\(([^)]*)\)(.*)$', s)
    suffix = ""
    if m:
        type_list = m.group(1)
        suffix = m.group(2)
    else:
        type_list = s
    type_list = list(map(lambda x: x.strip(" ") + suffix, re.split(r'[,/]', type_list)))
    return type_list


if __name__ == "__main__":

    flywire = Flywire.local()
    mcns = Neuprint.male_cns()

    # Load male and female cell types
    mcns_cell_types = mcns.cell_types
    flywire_cell_types = flywire.cell_types

    # Get male neuron types
    all_ids = np.concat(list(mcns_cell_types.values()))
    neuron_df = fetch_neurons(NC(bodyId=[int(cell_id) for cell_id in all_ids]), omit_rois=True)

    # Get flywire types
    flywire_primary_types = list(set(flywire_cell_types.keys()))
    flywire_secondary_types = [str(s) for s in flywire.db.cell_types["additional_type(s)"].unique()]
    flywire_secondary_types = [reformat_type_string(s) for s in flywire_secondary_types]
    flywire_secondary_types = [x for sublst in flywire_secondary_types for x in sublst if x != "nan"]
    flywire_all_types = list(set(flywire_primary_types + flywire_secondary_types))
    # Map flywire secondary type to primary type and vice versa
    flywire2_to_flywire1 = {}
    for primary_type, df in flywire.db.cell_types.groupby("primary_type"):
        secondary_types = df["additional_type(s)"].unique()
        secondary_types = [reformat_type_string(str(s)) for s in secondary_types]
        secondary_types = [x for sublst in secondary_types for x in sublst if x != "nan"]
        for st in secondary_types:
            if st in flywire2_to_flywire1:
                flywire2_to_flywire1[st].append(primary_type)
            else:
                flywire2_to_flywire1[st] = [primary_type]
    flywire2_to_flywire1 = {k: list(set(vals)) for (k, vals) in flywire2_to_flywire1.items()}

    # ==============
    # MALE TO FEMALE
    # ==============

    mcns_to_flywire = {}

    for mct, df in neuron_df.groupby("type"):
        matches = [reformat_type_string(str(s)) for s in df["flywireType"].unique()]
        matches = [x for sublst in matches for x in sublst]

        matched_fcts = []
        for fct in matches:
            if fct == "None":
                matched_fcts.append("_unmatched")
            if fct in flywire_primary_types:
                matched_fcts.append(fct)
            matched_fcts.extend(flywire2_to_flywire1.get(fct, []))
        if len(matched_fcts) == 0:  # haven't found any matches
            # Check if male name matches female name
            if mct in flywire_primary_types:
                matched_fcts.append(mct)
            matched_fcts.extend(flywire2_to_flywire1.get(mct, []))
            # Check if short male name matches female name
            mct_short, *suffix = mct.split("_")
            if mct_short in flywire_primary_types:
                matched_fcts.append(mct_short)
            matched_fcts.extend(flywire2_to_flywire1.get(mct_short, []))
            # Check if matches to slightly longer name
            try:
                mct_short = mct_short + "_" + suffix[0][0]
                fcts = list(filter(lambda x: x.startswith(mct_short), flywire_all_types))
                for fct in fcts:
                    if fct in flywire_primary_types:
                        matched_fcts.append(fct)
                    matched_fcts.extend(flywire2_to_flywire1.get(fct, []))
            except IndexError:
                pass

        if len(matched_fcts) == 0:  # still haven't found any matches -> phantom type
            matched_fcts.append(f"_unmatched")

        matched_fcts = list(set(matched_fcts))
        assert len(matched_fcts)

        mcns_to_flywire[mct] = matched_fcts

    mcns_to_flywire["_unmatched"] = []

    # ==============
    # FEMALE TO MALE
    # ==============

    flywire_to_mcns = {}

    for mct, fcts in mcns_to_flywire.items():
        for fct in fcts:
            if fct not in flywire_to_mcns:
                flywire_to_mcns[fct] = [mct]
            else:
                flywire_to_mcns[fct].append(mct)
    flywire_to_mcns = {fct: list(set(mcts)) for (fct, mcts) in flywire_to_mcns.items()}

    for fct in set(flywire_primary_types).difference(set(flywire_to_mcns.keys())):
        flywire_to_mcns[fct] = ["_unmatched"]
        mcns_to_flywire["_unmatched"].append(fct)
