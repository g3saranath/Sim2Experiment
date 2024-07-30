# Sim2Experiment
AIMD simulations Trained model - to identify stable structures to be reused in experiments

Dataset can be downloaded from [here](https://urldefense.com/v3/__https://drive.google.com/file/d/1BI6hA4sUEL3I037y7WcUmtIWIF6lPxjj/view?usp=sharing__;!!NpxR!j-3vgg4H4UNiawf12dwcBseYBSZc-kB1GHbGyiM2dDVXE3EGG3gEyJNMWGlqwdAyQSQFPR0GSEqyC1tbJjFV4yA7-sOYk5U$)

## Unzipping the data:
```
gzip -d filename.gz
tar -xvf to_share_HP_defects_data.tar
```
The datafiles will be present in /to_share_HP_defects


## Creating Env
```
conda create -n aimd
conda activate aimd
conda install python==3.10
pip install -r requirements.txt
python -m ipykernel install --user --name aimd
```

## Downloading required modules
```
wget https://github.com/ziatdinovmax/AtomicImageSimulator/archive/master.zip
unzip master.zip
```
