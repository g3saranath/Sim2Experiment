# Sim2Experiment
AIMD simulations Trained model - to identify stable structures to be reused in experiments

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
