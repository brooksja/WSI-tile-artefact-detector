# Attempt at creating an anomaly detector to flag WSI tiles with artefacts/weird colours/etc

from model import Artefact_detector
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import random_split,DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import os
import PIL.Image as im

torch.set_float32_matmul_precision('medium')

transform = T.Compose([
    T.ToTensor(),
    T.Resize(225,antialias=True),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(
    root = '/mnt/ravenclaw/Artefact_dataset/',
    transform=transform
)

train,valid = random_split(dataset,[0.8,0.2])
train_loader = DataLoader(train,batch_size=64,num_workers=os.cpu_count(),shuffle=True)
valid_loader = DataLoader(valid,batch_size=64,num_workers=os.cpu_count())

model = Artefact_detector()
trainer = pl.Trainer(accelerator='gpu',max_epochs=100,callbacks=[EarlyStopping(monitor='valid_loss',min_delta=1e-4,patience=5)],log_every_n_steps=10)
trainer.fit(model,train_loader,valid_loader)

weights = model.load_default_weights()
model = Artefact_detector.load_from_checkpoint(weights)
model.eval()
transform = model.default_transforms()
print(model(transform(im.open('/mnt/ravenclaw/Artefact_dataset/Artefact/M227_(14495,38847).jpg'))))