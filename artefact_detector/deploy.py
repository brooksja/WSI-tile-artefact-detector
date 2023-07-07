from helpers import quantify_image
import argparse
import pickle
import cv2

def run_artefact_detector(image,model = 'artefact_detector.model'):
	# load the anomaly detection model
	model = pickle.loads(open(model, "rb").read())
	# load the input image, convert it to the HSV color space, and
	# quantify the image in the *same manner* as we did during training
	image = cv2.imread(image)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	features = quantify_image(hsv, bins=(4,6,3))

	# use the anomaly detector model and extracted features to determine
	# if the example image is an anomaly or not
	preds = model.predict([features])[0]
	# returns preds = -1 if anomaly, 1 if normal
	return preds