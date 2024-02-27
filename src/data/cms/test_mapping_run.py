file_name = "/afs/cern.ch/user/r/rtu/CMSSW_11_1_0_pre5_PY3/src/PhysicsTools/NanoAODTools/tmp/2017haronicTau/QCD_HT2000toInf_TuneCP5_13TeV-madgraphMLM-pythia8/2C123F00-EE3E-2B46-9A39-5EAF8C20A2D3_Skim.root"

mappings = {'HHHTo4B2Tau_SM' : 1, 
            'GluGluToHHHTo6B_SM' : 2,
            'GluGluToHHTo4B'  : 3,
            'GluGluToHHTo2B2Tau': 4,
            'QCD_HT': 5,
            'TT_Mtt': 6,
            'TTToSemiLeptonic': 7,
            'TTToHadronic_': 6,
            'WJets': 8,
            'ZJets': 9,
            'WWTo4Q': 10,
            'WWW_4F': 10,
            'WWZ_4F': 10,
            'WZZ_': 10,
            'ZZTo4Q': 10,
            'ZZZ_': 10,
}

for key,value in mappings.items():
    if file_name.find(key) != -1:
        a = value
        break
    else:
        a = 0
print(a)