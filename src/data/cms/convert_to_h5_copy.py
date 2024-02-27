import logging
from pathlib import Path

import awkward as ak
import click
import h5py
import numpy as np
import uproot
import vector
from coffea.nanoevents import BaseSchema, NanoEventsFactory

vector.register_awkward()

logging.basicConfig(level=logging.INFO)

N_JETS = 10
N_FJETS = 3
N_MASSES = 45
MIN_JET_PT = 20
MIN_FJET_PT = 200
MIN_JETS = 4
N_TAU = 6
MIN_MASS = 50
PROJECT_DIR = Path(__file__).resolve().parents[3]
'''
mappings = {'HHHTo4B2Tau_SM' : 1, 
            'GluGluToHHHTo6B_SM' : 2,
            'GluGluToHHTo4B'  : 3,
            'GluGluToHHTo2B2Tau': 4,
            'QCD_HT': 5,
            'TT_Mtt': 6,
            'TTToSemiLeptonic': 6,
            'TTToHadronic_': 7,
            'WJets': 8,
            'ZJets': 9,
            'WWTo4Q': 10,
            'WWW_4F': 10,
            'WWZ_4F': 10,
            'WZZ_': 10,
            'ZZTo4Q': 10,
            'ZZZ_': 10,
}
'''
mappings = {'HHHTo4B2Tau' : 1, 
            'GluGluToHHHTo6B' : 2,
            'GluGluToHHTo4B'  : 3,
            'GluGluToHHTo2B2Tau': 4,
            'QCDHT': 5,
            'TTMtt': 6,
            'TTToSemiLeptonic': 6,
            'TTToHadronic': 7,
            'WJets': 8,
            'ZJets': 9,
            'WWTo4Q': 10,
            'WWW4F': 10,
            'WWZ4F': 10,
            'WZZ': 10,
            'ZZTo4Q': 10,
            'ZZZ': 10,
}
def get_n_features(name, events, iterator):
    if name.format(i=iterator[0]) not in dir(events):
        logging.warning(f"Variable {name.format(i=iterator[0])} does not exist in tree; returning all 0s")
        return ak.from_numpy(np.zeros((len(events), len(iterator))))
    return ak.concatenate(
        [np.expand_dims(events[name.format(i=i)], axis=-1) for i in iterator],
        axis=-1,
    )


def get_datasets(events):
    # small-radius jet info
    pt = get_n_features("jet{i}Pt", events, range(1, N_JETS + 1))
    #ptcorr = get_n_features("jet{i}PtCorr", events, range(1, N_JETS + 1))
    eta = get_n_features("jet{i}Eta", events, range(1, N_JETS + 1))
    phi = get_n_features("jet{i}Phi", events, range(1, N_JETS + 1))
    btag = get_n_features("jet{i}DeepFlavB", events, range(1, N_JETS + 1))
    #jet_id = get_n_features("jet{i}JetId", events, range(1, N_JETS + 1))
    #higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, range(1, N_JETS + 1))
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, range(1, N_JETS + 1))
    #matched_fj_idx = get_n_features("jet{i}FatJetMatchedIndex", events, range(1, N_JETS + 1))
    inv_mass = get_n_features("jet{i}Mass", events, range(1, N_JETS + 1))

    # paired masses
    #mass = get_n_features("mass{i}", events, range(N_MASSES))


    # keep events with >= MIN_JETS small-radius jets
    mask = ak.num(pt[pt > MIN_JET_PT]) >= MIN_JETS
    pt = pt[mask]
    #ptcorr = ptcorr[mask]
    eta = eta[mask]
    phi = phi[mask]
    btag = btag[mask]
    #mass = mass[mask]
    hadron_flavor = hadron_flavor[mask]
    inv_mass = inv_mass[mask]

    #mass = mass[mask]
    #tau info
    tau_pt = get_n_features("tau{i}Pt", events, range(1, N_TAU + 1))
    tau_mask = tau_pt > 20
    tau_mass = get_n_features("tau{i}Mass", events, range(1, N_TAU + 1))
    tau_eta = get_n_features("tau{i}Eta", events, range(1, N_TAU + 1))
    tau_phi = get_n_features("tau{i}Phi", events, range(1, N_TAU + 1))
    tau_rawDeepTau2017v2p1VSjet = get_n_features("tau{i}rawDeepTau2017v2p1VSjet", events, range(1, N_TAU + 1))

    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT
    #mask_mass = mass > MIN_MASS

    # mask to define zero-padded large-radius jets
    #fj_mask = fj_pt > MIN_FJET_PT

    # require hadron_flavor == 5 (i.e. b-jet ghost association matching)
    #higgs_idx = ak.where(higgs_idx != 0, ak.where(hadron_flavor == 5, higgs_idx, -1), 0)

    # index of small-radius jet if Higgs is reconstructed
    #h1_bs = ak.local_index(higgs_idx)[higgs_idx == 1]
    #h2_bs = ak.local_index(higgs_idx)[higgs_idx == 2]
    #h3_bs = ak.local_index(higgs_idx)[higgs_idx == 3]
    signal = ak.from_numpy(np.full(len(events),events.signal,dtype = int))

    # check/fix small-radius jet truth (ensure max 2 small-radius jets per higgs)
    '''
    check = (
        np.unique(ak.count(h1_bs, axis=-1)).to_list()
        + np.unique(ak.count(h2_bs, axis=-1)).to_list()
        + np.unique(ak.count(h3_bs, axis=-1)).to_list()
    )
    if 3 in check:
        logging.warning("some Higgs bosons match to 3 small-radius jets! Check truth")
    '''
    h1_bs = get_n_features("h1_t3_match{i}", events, range(1, 3))
    h2_bs = get_n_features("h2_t3_match{i}", events, range(1, 3))
    h3_bs = get_n_features("higgs3_tau{i}", events, range(1, 3))

    merged_array = ak.concatenate([h1_bs, h2_bs], axis=1)
    merged_array = merged_array.to_numpy()
    '''
    has_duplicates = np.any((merged_array != -1) & (merged_array[:, None, :] == merged_array[:, :, None]), axis=1)
    merged_array[has_duplicates] = -1
    '''
    def unique_row(row):
        unique_values = np.unique(row[row != -1])
        if len(unique_values) != len(row[row != -1]):
            return np.full_like(row, -1)
        return row

    unique_row_vectorized = np.vectorize(unique_row, signature='(n)->(n)')

    merged_array = unique_row_vectorized(merged_array)
    

    Mask_list = [np.sum(row) for row in mask.to_numpy()]
    merged_array = np.where(merged_array >= np.array(Mask_list).reshape(-1, 1), -1, merged_array)
    
    '''
    h1_bs = ak.fill_none(ak.pad_none(h1_bs, 2, clip=True), -1)
    h2_bs = ak.fill_none(ak.pad_none(h2_bs, 2, clip=True), -1)
    h3_bs = ak.fill_none(ak.pad_none(h3_bs, 2, clip=True), -1)
    

    h1_bb = ak.fill_none(ak.pad_none(h1_bb, 1, clip=True), -1)
    h2_bb = ak.fill_none(ak.pad_none(h2_bb, 1, clip=True), -1)
    h3_bb = ak.fill_none(ak.pad_none(h3_bb, 1, clip=True), -1)
    '''

    #h1_b1, h1_b2 = h1_bs[:, 0], h1_bs[:, 1]
    #h2_b1, h2_b2 = h2_bs[:, 0], h2_bs[:, 1]
    h1_b1, h1_b2 = merged_array[:, 0], merged_array[:, 1]
    h2_b1, h2_b2 = merged_array[:, 2], merged_array[:, 3]
    h3_b1, h3_b2 = h3_bs[:, 0], h3_bs[:, 1]

    '''
    # mask whether Higgs can be reconstructed as 2 small-radius jet
    h1_mask = ak.all(h1_bs != -1, axis=-1)
    h2_mask = ak.all(h2_bs != -1, axis=-1)
    h3_mask = ak.all(h3_bs != -1, axis=-1)

    # mask whether Higgs can be reconstructed as 1 large-radius jet
    h1_fj_mask = ak.all(h1_bb != -1, axis=-1)
    h2_fj_mask = ak.all(h2_bb != -1, axis=-1)
    h3_fj_mask = ak.all(h3_bb != -1, axis=-1)
    '''

    datasets = {}
    datasets["INPUTS/Jets/MASK"] = mask.to_numpy()
    datasets["INPUTS/Jets/pt"] = pt.to_numpy()
    #datasets["INPUTS/Jets/ptcorr"] = ptcorr.to_numpy()
    datasets["INPUTS/Jets/eta"] = eta.to_numpy()
    datasets["INPUTS/Jets/phi"] = phi.to_numpy()
    datasets["INPUTS/Jets/sin_phi"] = np.sin(phi.to_numpy())
    datasets["INPUTS/Jets/cos_phi"] = np.cos(phi.to_numpy())
    datasets["INPUTS/Jets/btag"] = btag.to_numpy()
    datasets["INPUTS/Jets/mass"] = inv_mass.to_numpy()
    #datasets["INPUTS/Jets/jetid"] = jet_id.to_numpy()
    #datasets["INPUTS/Jets/matchedfj"] = matched_fj_idx.to_numpy()
    #datasets["INPUTS/Jets/invmass"] = inv_mass.to_numpy()

    # Taus
    datasets["INPUTS/Taus/MASK"] = tau_mask.to_numpy()
    datasets["INPUTS/Taus/pt"] = tau_pt.to_numpy()
    datasets["INPUTS/Taus/mass"] = tau_mass.to_numpy()
    datasets["INPUTS/Taus/eta"] = tau_eta.to_numpy()
    datasets["INPUTS/Taus/phi"] = tau_phi.to_numpy()
    datasets["INPUTS/Taus/sin_phi"] = np.sin(tau_phi.to_numpy())
    datasets["INPUTS/Taus/cos_phi"] = np.cos(tau_phi.to_numpy())
    datasets["INPUTS/Taus/rawDeepTau2017v2p1VSjet"] = tau_rawDeepTau2017v2p1VSjet.to_numpy()

    #global (Met and Ht)
    met = events.met
    ht = events.ht
    ntau = events.ntau

    datasets["INPUTS/MET/met"] = met.to_numpy()
    datasets["INPUTS/HT/ht"] = ht.to_numpy()
    datasets["INPUTS/HT/ntau"] = ntau.to_numpy()

    #for i in range(0, N_MASSES):
    #    datasets[f"INPUTS/Masses/MASK{i}"] = mask_mass.to_numpy()[:, i]
    #    datasets[f"INPUTS/Masses/mass{i}"] = mass.to_numpy()[:, i]

    #datasets["TARGETS/h1/mask"] = h1_mask.to_numpy()
    datasets["TARGETS/h1/b1"] = h1_b1#.to_numpy()
    datasets["TARGETS/h1/b2"] = h1_b2#.to_numpy()

    #datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1#.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2#.to_numpy()

    #datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
    datasets["TARGETS/lh/tau1"] = h3_b1.to_numpy()
    datasets["TARGETS/lh/tau2"] = h3_b2.to_numpy()
    datasets["CLASSIFICATIONS/EVENT/signal"] = signal.to_numpy()

    '''
    datasets["TARGETS/bh1/mask"] = h1_fj_mask.to_numpy()
    datasets["TARGETS/bh1/bb"] = h1_bb.to_numpy().reshape(h1_fj_mask.to_numpy().shape)

    datasets["TARGETS/bh2/mask"] = h2_fj_mask.to_numpy()
    datasets["TARGETS/bh2/bb"] = h2_bb.to_numpy().reshape(h2_fj_mask.to_numpy().shape)

    datasets["TARGETS/bh3/mask"] = h3_fj_mask.to_numpy()
    datasets["TARGETS/bh3/bb"] = h3_bb.to_numpy().reshape(h3_fj_mask.to_numpy().shape)
    '''

    return datasets


@click.command()
@click.argument("in-files", nargs=-1)
@click.option("--out-file", default=f"{PROJECT_DIR}/data/cms/hhh_training.h5", help="Output file.")
@click.option("--train-frac", default=0.95, help="Fraction for training.")
def main(in_files, out_file, train_frac):
    all_datasets = {}
    for file_name in in_files:
        print(file_name)
        if 'JetHT' in file_name: continue
        if 'SingleMuon' in file_name: continue
        if 'BTagCSV' in file_name: continue

        with uproot.open(file_name) as in_file:
            num_entries = in_file["Events"].num_entries
            if "training" in out_file:
                entry_start = None
                entry_stop = int(train_frac * num_entries)
            else:
                entry_start = int(train_frac * num_entries)
                entry_stop = None
            events = NanoEventsFactory.from_root(
                in_file,
                treepath="Events",
                entry_start=entry_start,
                entry_stop=entry_stop,
                schemaclass=BaseSchema,
            ).events()
            #if 'GluGluToHHHTo6B_SM' in file_name:
            #    events.signal = 1
            #else:
            #    events.signal = 0
            for key,value in mappings.items():
                if key in file_name:
                    events.signal = value
                    break
                else:
                    events.signal = 0

            datasets = get_datasets(events)
            for dataset_name, data in datasets.items():
                if dataset_name not in all_datasets:
                    all_datasets[dataset_name] = []
                all_datasets[dataset_name].append(data)



    print(all_datasets.keys())
    with h5py.File(out_file, "w") as output:

        for dataset_name, all_data in all_datasets.items():
            concat_data = np.concatenate(all_data, axis=0)
            output.create_dataset(dataset_name, data=concat_data)


    # shuffle randomly
    data = h5py.File(out_file, 'r')


    with h5py.File(out_file.replace('.h5','_random.h5'), 'w') as out:
        print(data)
        indexes = np.arange(data['CLASSIFICATIONS/EVENT/signal'].shape[0])
        np.random.shuffle(indexes)
        for key in all_datasets.keys():
            print(key)
            feed = np.take(np.array(data[key]), indexes, axis=0)
            out.create_dataset(key, data=feed)


if __name__ == "__main__":
    main()
