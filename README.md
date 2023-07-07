# WSI-tile-artefact-detector
Small model designed to catch artefacts in WSI tiles

## Usage
To use the detector within a script: 
  1. pip install the repo
  2. add: from artefact_detector.deploy import run_artefact_detector to your python script
  3. call run_artefact_detector(tile,model); where tile is a tile from a WSI and model is either None (for default) or a path to a model created by the user

## Training new models
To train a new detector model:
  1. Clone the repo
  2. cd into the repo
  3. run python artefact_detector.train.py -d PATH/TO/IMAGE/DATASET -m PATH/TO/SAVE/MODEL/TO
