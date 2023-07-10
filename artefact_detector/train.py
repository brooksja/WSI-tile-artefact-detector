# Attempt at creating an anomaly detector to flag WSI tiles with artefacts/weird colours/etc

# Following https://pyimagesearch.com/2020/01/20/intro-to-anomaly-detection-with-opencv-computer-vision-and-scikit-learn/

from helpers import load_dataset
from sklearn.ensemble import IsolationForest
import argparse
import pickle
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output anomaly detection model")
args = vars(ap.parse_args())

# load and quantify our image dataset
print("[INFO] preparing dataset...")
data = load_dataset(args["dataset"], bins=(4,6,3))
# train the anomaly detection model
print("[INFO] fitting anomaly detection model...")
model = IsolationForest(n_estimators=100,random_state=42)
model.fit(data)

# serialize the anomaly detection model to disk
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
