# This test script also serves as an example of how to include the detector in other scripts
# model output < 0.5 => artefact

from model import Artefact_detector
import PIL.Image as im

model = Artefact_detector()
weights = model.load_default_weights()
model = Artefact_detector.load_from_checkpoint(weights)
model.eval().cpu()
transform = model.default_transforms()
print(model.model(transform(im.open('/mnt/ravenclaw/LLOVET-LIVER-HCC/Resections/patches_5X/M227/M227_(13915,34788).jpg')).unsqueeze(0)).detach().squeeze().numpy())