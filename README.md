# hhh
The framework is based on https://github.com/ucsd-hep-ex/hhh.

## 1. Pull and start the Docker container(On lxplus, we use singularity)
```bash
docker pull jmduarte/hhh
singularity shell --bind /eos/user/r/rtu:/outdir1 --bind /afs/cern.ch/user/r/rtu:/outputdir /eos/user/r/rtu/hhh_latest.sif
```

## 2. Check out the GitHub repository
```bash
cd work
git@github.com:RuobingTu/hhh_spanet_conversion.git
```

## 3. Install the Python package(s)
```bash
cd hhh
pip install -e .
cd ..
```

## 4. Convert the dataset(s)
Copy the ROOT TTree datasets from:
- CERN EOS: `/eos/user/r/rtu/public/inputs-2017/inclusive-weights/HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree.root`


Convert to training and testing HDF5 files.
```bash
python -m src.data.cms.convert_to_h5_jetpair_PNet.py HHHTo4B2Tau_c3_0_d4_0_TuneCP5_13TeV-amcatnlo-pythia8_tree.root --out-file hhh_training.h5
```

## 5. Run the SPANet training
Override options file with `--gpus 0` if no GPUs are available.
```bash
python -m spanet.train -of options_files/cms/HHH_4b2tau_classification.json [--gpus 0]
```
you need to modify the input file and event information path like below.
```
    "event_info_file": "/hpcfs/cms/cmsgpu/turuobing/HHH_4b2tau_classification.yaml",
    "training_file": "/hpcfs/cms/cmsgpu/turuobing/HHH_QurdJetTrigger_jetpair_PNet_training_random.h5",
    "trial_output_dir": "/hpcfs/cms/cmsgpu/turuobing/HHH_output_PNet_classification_new",
```

since I add the classification part in 'HHH_4b2tau_classification.json', if we don't add any background, we need to modify the hyperparameter to zero.
```
  "classification_loss_scale": 0,
```

## 6. Evaluate the SPANet training
Assuming the output log directory is `spanet_output/version_0`.
Add `--gpu` if a GPU is available.
```bash
python -m spanet.test spanet_output/version_0 -tf hhh_testing.h5 [--gpu]
```


# Instructions for CMS data set baseline(It's Marko's early work, has done before 2023/11)
The CMS dataset was updated to run with the `v26` setup (`nAK4 >= 4 and HLT selection`). The update includes the possibility to apply the b-jet energy correction. By keeping events with at a least 4 jets, the boosted training can be performed on a maximum number of events and topologies.

List of samples (currently setup validated using 2018):
```
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016APV.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2017.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root
```

To run the framework, first convert the samples (this will allow to use both jets `pt` or `ptcorr`, steerable from the configuration file:
```
mkdir data/cms/v26/
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_training.h5
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_testing.h5
```

Then training can be done via:

```
python -m spanet.train -of options_files/cms/hhh_v26.json --gpus 1
```

Two config files exist for the event options:
```
event_files/cms/hhh.yaml # regular jet pT
event_files/cms/hhh_bregcorr.yaml # jet pT with b-jet energy correction scale factors applied
```

Note: to run the training with the b-jet energy correction applied, the `log_normalize` of the input variable was removed. Keeping it caused a 'Assignement collision'.
