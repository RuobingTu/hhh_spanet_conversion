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

def get_n_features(name, events, iterator):
    if name.format(i=iterator[0]) not in dir(events):
        logging.warning(f"Variable {name.format(i=iterator[0])} does not exist in tree; returning all 0s")
        return ak.from_numpy(np.zeros((len(events), len(iterator))))
    return ak.concatenate(
        [np.expand_dims(events[name.format(i=i)], axis=-1) for i in iterator],
        axis=-1,
    )

a = 5

# in_file = "/eos/user/r/rtu/pieces/GluGluToHHHTo4B2TauSM/GluGluToHHHTo4B2TauSM.rootQuadPFJetTrigger"
# in_file_name = "/outdir1/pieces/GluGluToHHHTo4B2TauSM/GluGluToHHHTo4B2TauSM.rootQuadPFJetTrigger"
if a == 1:
   in_file_name =  "/outputdir/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/336E_15_Skim.root"
if a==2:
    in_file_name = "/outputdir/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/nano_6_Skim.rootXbb0.5"
if a==3:
    in_file_name = "/outputdir/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/nano_106_Skim.root"
if a==4:
    in_file_name = "/outputdir/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/nano_106_Skim.root"
if a==5:
    in_file_name = "/outputdir/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/nano_106_Skim.root"

in_file = uproot.open(in_file_name)
num_entries = in_file["Events"].num_entries
events = NanoEventsFactory.from_root(
    in_file,
    treepath="Events",
    entry_start= None,
    entry_stop=num_entries,
    schemaclass=BaseSchema,
).events()

if a == 1:
    tree = in_file["Events"]
    tau_pt = get_n_features("tau{i}idDeepTau2017v2p1VSjet", events, range(1, 7))
    h3_bs = get_n_features("higgs3_tau{i}", events, range(1, 3))
    tau_pt_list = tau_pt.to_numpy().flatten().tolist()
    Matched_tau_pt = []
    for i in range(num_entries):
        for j in h3_bs[i]:
            Matched_tau_pt.append(tau_pt[i][j])

if a ==2:
    tree = in_file["Events"]
    bh_mass = get_n_features("bh{i}_t3_mass", events, range(1, 3))
    bjetmass = get_n_features("fatJet{i}Mass", events, range(1, 5))
    bh_mass_list = bh_mass.to_numpy().flatten().tolist()
    bjetmass_list = bjetmass.to_numpy().flatten().tolist()

if a ==5:
    tree = in_file["Events"]
    th_mass = get_n_features("higgs{i}_mass", events, range(3,4))
    th_mass_list = th_mass.to_numpy().flatten().tolist()

if a == 3:
    tree = in_file["Events"]
    rh_mass = get_n_features("h{i}_t3_mass", events, range(1, 3))
    rh_massRMoverlap = get_n_features("rh{i}_t3_mass", events, range(1, 3))
    rh_mass_list = rh_mass.to_numpy().flatten().tolist()
    rh_massRMoverlap_list = rh_massRMoverlap.to_numpy().flatten().tolist()

if a == 4:
    tree = in_file["Events"]
    fatjet_Xbb = get_n_features("fatJet{i}PNetXbb", events, range(1, 5))
    fatjet_matched_tag = get_n_features("fatJet{i}HiggsMatched", events, range(1, 5))
    fatjet_Xbb_matched = ak.Array(fatjet_Xbb[fatjet_matched_tag])
    fatjet_Xbb = fatjet_Xbb.to_numpy().flatten().tolist()
    fatjet_Xbb_matched = ak.flatten(fatjet_Xbb_matched).tolist()
    print(len([x for x in fatjet_Xbb_matched if x > 0.5])/len(fatjet_Xbb_matched))


import matplotlib.pyplot as plt

# 绘制直方图
# plt.hist(Matched_tau_pt, bins=40, alpha=0.5, label='Matched_tau', range=(1, 300))
# plt.hist(tau_pt_list, bins=40, alpha=0.5, label='loose_tau', range=(1, 300))
if a == 1:
    plt.hist(Matched_tau_pt, bins=270, alpha=0.5, label='Matched_tau', range=(1, 11), histtype='step')
    plt.hist(tau_pt_list, bins=270, alpha=0.5, label='loose_tau', range=(1, 11), histtype='step')
    # 添加图例
    plt.legend(loc='upper right')
    plt.xlabel('idDeepTau2017v2p1VSjet')
    plt.ylabel('Event/1')

    # 保存为png文件
    plt.savefig('/outputdir/hhh_spanet_convert/hhh/src/data/cms/histogram_tauid10.png')

if a == 2:
    plt.hist(bh_mass_list, bins=40, alpha=0.5, label='Matched_bh', range=(1, 300), histtype='step')
    plt.hist(bjetmass_list, bins=40, alpha=0.5, label='boostjet', range=(1, 300), histtype='step')
    # 添加图例
    plt.legend(loc='upper right')
    plt.xlabel('Mass/GeV')
    plt.ylabel('Event/1')

    # 保存为png文件
    plt.savefig('/outputdir/hhh_spanet_convert/hhh/src/data/cms/histogram_boostHmassOnlyTmp.png')

if a == 3:
    plt.hist(rh_mass_list, bins=40, alpha=0.5, label='rh_mass', range=(1, 300), histtype='step')
    plt.hist(rh_massRMoverlap_list, bins=40, alpha=0.5, label='rh_massRMOverlap', range=(1, 300), histtype='step')
    # 添加图例
    plt.legend(loc='upper right')
    plt.xlabel('Mass/GeV')
    plt.ylabel('Event/1')

    # 保存为png文件
    plt.savefig('/outputdir/hhh_spanet_convert/hhh/src/data/cms/histogram_rhMass4b2tau.png')

if a == 4:
    plt.hist(fatjet_Xbb, bins=16, alpha=0.5, label='fatjet_Xbb', range=(0.001, 1), histtype='step')
    plt.hist(fatjet_Xbb_matched, bins=16, alpha=0.5, label='fatjet_Xbb_matched', range=(0.001, 1), histtype='step')
    # 添加图例
    plt.legend(loc='upper right')
    plt.xlabel('Xbb_score')
    plt.ylabel('Event/1')

    # 保存为png文件
    plt.savefig('/outputdir/hhh_spanet_convert/hhh/src/data/cms/histogram_fatjetXbb0_2b2tau.png')

if a == 5:
    plt.hist(th_mass_list, bins=40, alpha=0.5, label='hTauTau_mass', range=(1, 300), histtype='step')
    # 添加图例
    plt.legend(loc='upper right')
    plt.xlabel('Mass/GeV')
    plt.ylabel('Event/1')

    # 保存为png文件
    plt.savefig('/outputdir/hhh_spanet_convert/hhh/src/data/cms/histogram_thMass4b2tau.png')
