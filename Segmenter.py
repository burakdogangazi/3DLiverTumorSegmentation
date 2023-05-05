from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from UNetModel import UNet
import torch
import pytorch_lightning as pl
import torchio as tio
import nibabel as nib
from celluloid import Camera
from IPython.display import HTML
import os
import tempfile


class Segmenter(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = UNet()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Train Loss", loss)
        if batch_idx % 50 == 0:
            self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Train")
        return loss
    
        
    def validation_step(self, batch, batch_idx):
        # You can obtain the raw volume arrays by accessing the data attribute of the subject
        img = batch["CT"]["data"]
        mask = batch["Label"]["data"][:,0]  # Remove single channel as CrossEntropyLoss expects NxHxW
        mask = mask.long()
        
        pred = self(img)
        loss = self.loss_fn(pred, mask)
        
        # Logs
        self.log("Val Loss", loss)
        self.log_images(img.cpu(), pred.cpu(), mask.cpu(), "Val")
        
        return loss

    
    def log_images(self, img, pred, mask, name):
        
        results = []
        pred = torch.argmax(pred, 1) # Take the output with the highest value
        axial_slice = 50  # Always plot slice 50 of the 96 slices
        
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(mask[0][:,:,axial_slice]==0, mask[0][:,:,axial_slice])
        axis[0].imshow(mask_, alpha=0.6)
        axis[0].set_title("Ground Truth")
        
        axis[1].imshow(img[0][0][:,:,axial_slice], cmap="bone")
        mask_ = np.ma.masked_where(pred[0][:,:,axial_slice]==0, pred[0][:,:,axial_slice])
        axis[1].imshow(mask_, alpha=0.6, cmap="autumn")
        axis[1].set_title("Pred")

        self.logger.experiment.add_figure(f"{name} Prediction vs Label", fig, self.global_step)

            
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.optimizer]
    


model = Segmenter.load_from_checkpoint("./segmentationmodel.ckpt",strict=False)
model.eval()

def transform_nii(nii_bytes):
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.nii.gz') as temp:
        temp.write(nii_bytes)
        temp.flush()

        # Load NIfTI file
        nii_obj = nib.load(temp.name)
        nii_data = nii_obj.get_fdata()

    # Data preprocessing
    process = tio.Compose([
        tio.CropOrPad((256, 256, 200)),
        tio.RescaleIntensity((-1, 1))
    ])

    # Process data
    processed_data = process(tio.ScalarImage(temp.name, affine=nii_obj.affine))

    # Convert data to tensor
    tensor = processed_data.data.to(torch.float32)

    # Delete temporary file
    os.remove(temp.name)

    return tensor



def get_result(tensor):
    fig = plt.figure()
    camera = Camera(fig)  # create the camera object from celluloid
    pred = tensor.argmax(0)

    for i in range(0, tensor.shape[3], 2):  # axial view
        plt.imshow(tensor[0, :, :, i], cmap="bone")
        mask_ = np.ma.masked_where(pred[:, :, i] == 0, pred[:, :, i])
        plt.imshow(mask_, alpha=0.1, cmap="autumn")
        camera.snap()  # Store the current slice
    animation = camera.animate()  # create the animation

    # convert the animation to HTML5 video format
    html_video = HTML(animation.to_html5_video())

    return html_video

