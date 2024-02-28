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
N_FJETS = 4
N_LEP = 4
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
            'HHHTo6B' : 2,
            'GluGluToHHTo4B'  : 3,
            'GluGluToHHTo2B2Tau': 4,
            'QCD_HT': 5,
            'TTTo2L2Nu': 6,
            'TTToSemiLeptonic': 7,
            'TTToHadronic': 8,
            'WJets': 9,
            'ZJets': 10,
            'WWTo': 11,
            'WZTo': 11,
            'ZZTo': 11,
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
    btagPN = get_n_features("jet{i}PNetB", events, range(1, N_JETS + 1))
    #jet_id = get_n_features("jet{i}JetId", events, range(1, N_JETS + 1))
    #higgs_idx = get_n_features("jet{i}HiggsMatchedIndex", events, range(1, N_JETS + 1))
    hadron_flavor = get_n_features("jet{i}HadronFlavour", events, range(1, N_JETS + 1))
    #matched_fj_idx = get_n_features("jet{i}FatJetMatchedIndex", events, range(1, N_JETS + 1))
    inv_mass = get_n_features("jet{i}Mass", events, range(1, N_JETS + 1))
    bregcorr = get_n_features("jet{i}bRegCorr", events, range(1, N_JETS + 1))

    FJpt = get_n_features("fatJet{i}Pt", events, range(1, N_FJETS + 1))
    FJeta = get_n_features("fatJet{i}Eta", events, range(1, N_FJETS + 1))
    FJphi = get_n_features("fatJet{i}Phi", events, range(1, N_FJETS + 1))
    FJPN_mass = get_n_features("fatJet{i}Mass", events, range(1, N_FJETS + 1))
    FJSD_mass = get_n_features("fatJet{i}MassSD_UnCorrected", events, range(1, N_FJETS + 1))
    FJ_Xbb = get_n_features("fatJet{i}PNetXbb", events, range(1, N_FJETS + 1))
    FJ_Xjj = get_n_features("fatJet{i}PNetXjj", events, range(1, N_FJETS + 1))
    eventWeight = get_n_features("{i}", events, ["eventWeight"])

    # paired masses
    #mass = get_n_features("mass{i}", events, range(N_MASSES))
    NumberOfHiggsMark = get_n_features("{i}", events, ['rh1_t3_match', 'rh2_t3_match', \
                                                       'bh1_t3_Matched', 'bh1_t3_Matched', 'bh1_t3_Matched', \
                                                        'bh2_t3_Matched', 'bh2_t3_Matched', 'bh2_t3_Matched', \
                                                        'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match', 'higgs3_tau_match',])
    NumberOfHiggs = ak.sum(NumberOfHiggsMark, axis=1)


    # keep events with >= MIN_JETS small-radius jets
    mask = ak.num(pt[pt > MIN_JET_PT]) >= 0
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

    lep_pt = get_n_features("lep{i}Pt", events, range(1, N_LEP + 1))
    lep_mask = lep_pt > 20
    lep_Id = get_n_features("lep{i}Id", events, range(1, N_LEP + 1))
    lep_eta = get_n_features("lep{i}Eta", events, range(1, N_LEP + 1))
    lep_phi = get_n_features("lep{i}Phi", events, range(1, N_LEP + 1))
  
    mass1 = get_n_features("massjet1jet{i}", events, range(2, N_JETS + 1))
    mass2 = get_n_features("massjet2jet{i}", events, range(3, N_JETS + 1))
    mass3 = get_n_features("massjet3jet{i}", events, range(4, N_JETS + 1))
    mass4 = get_n_features("massjet4jet{i}", events, range(5, N_JETS + 1))
    mass5 = get_n_features("massjet5jet{i}", events, range(6, N_JETS + 1))
    mass6 = get_n_features("massjet6jet{i}", events, range(7, N_JETS + 1))
    mass7 = get_n_features("massjet7jet{i}", events, range(8, N_JETS + 1))
    mass8 = get_n_features("massjet8jet{i}", events, range(9, N_JETS + 1))
    mass9 = get_n_features("massjet9jet{i}", events, range(10, N_JETS + 1))

    pt1 = get_n_features("ptjet1jet{i}", events, range(2, N_JETS + 1))
    pt2 = get_n_features("ptjet2jet{i}", events, range(3, N_JETS + 1))
    pt3 = get_n_features("ptjet3jet{i}", events, range(4, N_JETS + 1))
    pt4 = get_n_features("ptjet4jet{i}", events, range(5, N_JETS + 1))
    pt5 = get_n_features("ptjet5jet{i}", events, range(6, N_JETS + 1))
    pt6 = get_n_features("ptjet6jet{i}", events, range(7, N_JETS + 1))
    pt7 = get_n_features("ptjet7jet{i}", events, range(8, N_JETS + 1))
    pt8 = get_n_features("ptjet8jet{i}", events, range(9, N_JETS + 1))
    pt9 = get_n_features("ptjet9jet{i}", events, range(10, N_JETS + 1))

    eta1 = get_n_features("etajet1jet{i}", events, range(2, N_JETS + 1))
    eta2 = get_n_features("etajet2jet{i}", events, range(3, N_JETS + 1))
    eta3 = get_n_features("etajet3jet{i}", events, range(4, N_JETS + 1))
    eta4 = get_n_features("etajet4jet{i}", events, range(5, N_JETS + 1))
    eta5 = get_n_features("etajet5jet{i}", events, range(6, N_JETS + 1))
    eta6 = get_n_features("etajet6jet{i}", events, range(7, N_JETS + 1))
    eta7 = get_n_features("etajet7jet{i}", events, range(8, N_JETS + 1))
    eta8 = get_n_features("etajet8jet{i}", events, range(9, N_JETS + 1))
    eta9 = get_n_features("etajet9jet{i}", events, range(10, N_JETS + 1))

    phi1 = get_n_features("phijet1jet{i}", events, range(2, N_JETS + 1))
    phi2 = get_n_features("phijet2jet{i}", events, range(3, N_JETS + 1))
    phi3 = get_n_features("phijet3jet{i}", events, range(4, N_JETS + 1))
    phi4 = get_n_features("phijet4jet{i}", events, range(5, N_JETS + 1))
    phi5 = get_n_features("phijet5jet{i}", events, range(6, N_JETS + 1))
    phi6 = get_n_features("phijet6jet{i}", events, range(7, N_JETS + 1))
    phi7 = get_n_features("phijet7jet{i}", events, range(8, N_JETS + 1))
    phi8 = get_n_features("phijet8jet{i}", events, range(9, N_JETS + 1))
    phi9 = get_n_features("phijet9jet{i}", events, range(10, N_JETS + 1))

    dr1 = get_n_features("drjet1jet{i}", events, range(2, N_JETS + 1))
    dr2 = get_n_features("drjet2jet{i}", events, range(3, N_JETS + 1))
    dr3 = get_n_features("drjet3jet{i}", events, range(4, N_JETS + 1))
    dr4 = get_n_features("drjet4jet{i}", events, range(5, N_JETS + 1))
    dr5 = get_n_features("drjet5jet{i}", events, range(6, N_JETS + 1))
    dr6 = get_n_features("drjet6jet{i}", events, range(7, N_JETS + 1))
    dr7 = get_n_features("drjet7jet{i}", events, range(8, N_JETS + 1))
    dr8 = get_n_features("drjet8jet{i}", events, range(9, N_JETS + 1))
    dr9 = get_n_features("drjet9jet{i}", events, range(10, N_JETS + 1))
    
    # mask to define zero-padded small-radius jets
    mask = pt > MIN_JET_PT
    FJ_mask = FJpt > MIN_FJET_PT
    mask_mass1 = mass1 > 20
    mask_mass2 = mass2 > 20
    mask_mass3 = mass3 > 20
    mask_mass4 = mass4 > 20
    mask_mass5 = mass5 > 20
    mask_mass6 = mass6 > 20
    mask_mass7 = mass7 > 20
    mask_mass8 = mass8 > 20
    mask_mass9 = mass9 > 20
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
    h1_bs = get_n_features("rh1_t3_match{i}", events, range(1, 3))
    h2_bs = get_n_features("rh2_t3_match{i}", events, range(1, 3))
    FJ_bs = get_n_features("bh{i}_t3_fjidx", events, range(1, 3))
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
    merged_array = np.where(merged_array >= np.array(Mask_list).reshape(-1, 1), -1, merged_array)       #检测对应的jetidx是否超出上限
    
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
    h1_FJb1, h2_FJb2 = FJ_bs[:, 0], FJ_bs[:, 1]

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
    datasets["INPUTS/Jets/ptcorr"] = bregcorr.to_numpy()
    datasets["INPUTS/Jets/eta"] = eta.to_numpy()
    datasets["INPUTS/Jets/phi"] = phi.to_numpy()
    datasets["INPUTS/Jets/sin_phi"] = np.sin(phi.to_numpy())
    datasets["INPUTS/Jets/cos_phi"] = np.cos(phi.to_numpy())
    datasets["INPUTS/Jets/btag"] = btagPN.to_numpy()
    datasets["INPUTS/Jets/mass"] = inv_mass.to_numpy()
    #datasets["INPUTS/Jets/jetid"] = jet_id.to_numpy()
    #datasets["INPUTS/Jets/matchedfj"] = matched_fj_idx.to_numpy()
    #datasets["INPUTS/Jets/invmass"] = inv_mass.to_numpy()
    datasets["INPUTS/FJets/MASK"] = FJ_mask.to_numpy()
    datasets["INPUTS/FJets/pt"] = FJpt.to_numpy()
    datasets["INPUTS/FJets/eta"] = FJeta.to_numpy()
    datasets["INPUTS/FJets/phi"] = FJphi.to_numpy()
    datasets["INPUTS/FJets/sin_phi"] = np.sin(FJphi.to_numpy())
    datasets["INPUTS/FJets/cos_phi"] = np.cos(FJphi.to_numpy())
    datasets["INPUTS/FJets/Xbb"] = FJ_Xbb.to_numpy()
    datasets["INPUTS/FJets/Xjj"] = FJ_Xjj.to_numpy()
    datasets["INPUTS/FJets/SDmass"] = FJSD_mass.to_numpy()
    datasets["INPUTS/FJets/PNmass"] = FJPN_mass.to_numpy()

    # lepton(electron and muon)
    datasets["INPUTS/Lep/MASK"] = lep_mask.to_numpy()
    datasets["INPUTS/Lep/pt"] = lep_pt.to_numpy()
    datasets["INPUTS/Lep/eta"] = lep_eta.to_numpy()
    datasets["INPUTS/Lep/phi"] = lep_phi.to_numpy()
    datasets["INPUTS/Lep/sin_phi"] = np.sin(lep_eta.to_numpy())
    datasets["INPUTS/Lep/cos_phi"] = np.cos(lep_phi.to_numpy())
    datasets["INPUTS/Lep/Id"] = lep_Id.to_numpy()

    # jetpair
    datasets[f"INPUTS/Jet1/MASK"] = mask_mass1.to_numpy()
    datasets[f"INPUTS/Jet1/mass1"] = mass1.to_numpy()
    datasets[f"INPUTS/Jet1/pt1"] = pt1.to_numpy()
    datasets[f"INPUTS/Jet1/eta1"] = eta1.to_numpy()
    datasets[f"INPUTS/Jet1/phi1"] = phi1.to_numpy()
    datasets[f"INPUTS/Jet1/sinphi1"] = np.sin(phi1.to_numpy())
    datasets[f"INPUTS/Jet1/cosphi1"] = np.cos(phi1.to_numpy())
    datasets[f"INPUTS/Jet1/dr1"] = dr1.to_numpy()

    datasets[f"INPUTS/Jet2/MASK"] = mask_mass2.to_numpy()
    datasets[f"INPUTS/Jet2/mass2"] = mass2.to_numpy()
    datasets[f"INPUTS/Jet2/pt2"] = pt2.to_numpy()
    datasets[f"INPUTS/Jet2/eta2"] = eta2.to_numpy()
    datasets[f"INPUTS/Jet2/phi2"] = phi2.to_numpy()
    datasets[f"INPUTS/Jet2/sinphi2"] = np.sin(phi2.to_numpy())
    datasets[f"INPUTS/Jet2/cosphi2"] = np.cos(phi2.to_numpy())
    datasets[f"INPUTS/Jet2/dr2"] = dr2.to_numpy()

    datasets[f"INPUTS/Jet3/MASK"] = mask_mass3.to_numpy()
    datasets[f"INPUTS/Jet3/mass3"] = mass3.to_numpy()
    datasets[f"INPUTS/Jet3/pt3"] = pt3.to_numpy()
    datasets[f"INPUTS/Jet3/eta3"] = eta3.to_numpy()
    datasets[f"INPUTS/Jet3/phi3"] = phi3.to_numpy()
    datasets[f"INPUTS/Jet3/sinphi3"] = np.sin(phi3.to_numpy())
    datasets[f"INPUTS/Jet3/cosphi3"] = np.cos(phi3.to_numpy())
    datasets[f"INPUTS/Jet3/dr3"] = dr3.to_numpy()

    datasets[f"INPUTS/Jet4/MASK"] = mask_mass4.to_numpy()
    datasets[f"INPUTS/Jet4/mass4"] = mass4.to_numpy()
    datasets[f"INPUTS/Jet4/pt4"] = pt4.to_numpy()
    datasets[f"INPUTS/Jet4/eta4"] = eta4.to_numpy()
    datasets[f"INPUTS/Jet4/phi4"] = phi4.to_numpy()
    datasets[f"INPUTS/Jet4/sinphi4"] = np.sin(phi4.to_numpy())
    datasets[f"INPUTS/Jet4/cosphi4"] = np.cos(phi4.to_numpy())
    datasets[f"INPUTS/Jet4/dr4"] = dr4.to_numpy()

    datasets[f"INPUTS/Jet5/MASK"] = mask_mass5.to_numpy()
    datasets[f"INPUTS/Jet5/mass5"] = mass5.to_numpy()
    datasets[f"INPUTS/Jet5/pt5"] = pt5.to_numpy()
    datasets[f"INPUTS/Jet5/eta5"] = eta5.to_numpy()
    datasets[f"INPUTS/Jet5/phi5"] = phi5.to_numpy()
    datasets[f"INPUTS/Jet5/sinphi5"] = np.sin(phi5.to_numpy())
    datasets[f"INPUTS/Jet5/cosphi5"] = np.cos(phi5.to_numpy())
    datasets[f"INPUTS/Jet5/dr5"] = dr5.to_numpy()

    datasets[f"INPUTS/Jet6/MASK"] = mask_mass6.to_numpy()
    datasets[f"INPUTS/Jet6/mass6"] = mass6.to_numpy()
    datasets[f"INPUTS/Jet6/pt6"] = pt6.to_numpy()
    datasets[f"INPUTS/Jet6/eta6"] = eta6.to_numpy()
    datasets[f"INPUTS/Jet6/phi6"] = phi6.to_numpy()
    datasets[f"INPUTS/Jet6/sinphi6"] = np.sin(phi6.to_numpy())
    datasets[f"INPUTS/Jet6/cosphi6"] = np.cos(phi6.to_numpy())
    datasets[f"INPUTS/Jet6/dr6"] = dr6.to_numpy()

    datasets[f"INPUTS/Jet7/MASK"] = mask_mass7.to_numpy()
    datasets[f"INPUTS/Jet7/mass7"] = mass7.to_numpy()
    datasets[f"INPUTS/Jet7/pt7"] = pt7.to_numpy()
    datasets[f"INPUTS/Jet7/eta7"] = eta7.to_numpy()
    datasets[f"INPUTS/Jet7/phi7"] = phi7.to_numpy()
    datasets[f"INPUTS/Jet7/sinphi7"] = np.sin(phi7.to_numpy())
    datasets[f"INPUTS/Jet7/cosphi7"] = np.cos(phi7.to_numpy())
    datasets[f"INPUTS/Jet7/dr7"] = dr7.to_numpy()

    datasets[f"INPUTS/Jet8/MASK"] = mask_mass8.to_numpy()
    datasets[f"INPUTS/Jet8/mass8"] = mass8.to_numpy()
    datasets[f"INPUTS/Jet8/pt8"] = pt8.to_numpy()
    datasets[f"INPUTS/Jet8/eta8"] = eta8.to_numpy()
    datasets[f"INPUTS/Jet8/phi8"] = phi8.to_numpy()
    datasets[f"INPUTS/Jet8/sinphi8"] = np.sin(phi8.to_numpy())
    datasets[f"INPUTS/Jet8/cosphi8"] = np.cos(phi8.to_numpy())
    datasets[f"INPUTS/Jet8/dr8"] = dr8.to_numpy()

    datasets[f"INPUTS/Jet9/MASK"] = mask_mass9.to_numpy()
    datasets[f"INPUTS/Jet9/mass9"] = mass9.to_numpy()
    datasets[f"INPUTS/Jet9/pt9"] = pt9.to_numpy()
    datasets[f"INPUTS/Jet9/eta9"] = eta9.to_numpy()
    datasets[f"INPUTS/Jet9/phi9"] = phi9.to_numpy()
    datasets[f"INPUTS/Jet9/sinphi9"] = np.sin(phi9.to_numpy())
    datasets[f"INPUTS/Jet9/cosphi9"] = np.cos(phi9.to_numpy())
    datasets[f"INPUTS/Jet9/dr9"] = dr9.to_numpy()
    
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

    datasets["TARGETS/bh1/FJb"] = h1_FJb1.to_numpy()
    #datasets["TARGETS/h2/mask"] = h2_mask.to_numpy()
    datasets["TARGETS/h2/b1"] = h2_b1#.to_numpy()
    datasets["TARGETS/h2/b2"] = h2_b2#.to_numpy()

    datasets["TARGETS/bh2/FJb"] = h2_FJb2.to_numpy()

    #datasets["TARGETS/h3/mask"] = h3_mask.to_numpy()
    datasets["TARGETS/lh/tau1"] = h3_b1.to_numpy()
    datasets["TARGETS/lh/tau2"] = h3_b2.to_numpy()
    datasets["CLASSIFICATIONS/EVENT/signal"] = NumberOfHiggs.to_numpy()
    datasets["CLASSIFICATIONS/EVENT/classfication_signal"] = signal.to_numpy()
    datasets["EVENTBaseInfo/Weight"] = eventWeight.to_numpy()

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
