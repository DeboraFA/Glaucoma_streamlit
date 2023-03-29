

# ! pip install segmentation_models_pytorch
# ! pip install pytorch_lightning
# ! pip install torchvision
# ! pip install albumentations


import os
import torch
from pprint import pprint
from torch.utils.data import Dataset, DataLoader
import cv2
import random
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imutils
from skimage import measure
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


# # Load dataset

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# # Train U-Net

# In[ ]:


class GlaucomaModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, loss_name, lr, beta,  **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        self.lr = lr
        
        # for image segmentation dice loss could be the best first choice
        
        if loss_name == 'Dice':
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        if loss_name == 'Tversky':
            self.loss_fn = smp.losses.TverskyLoss(smp.losses.BINARY_MODE, from_logits=True, beta=beta)
        if loss_name == 'Lovasz':
            self.loss_fn = smp.losses.LovaszLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        loss = outputs[0]["loss"]

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
            f"{stage}_loss": loss,
            
        }
        
        self.log_dict(metrics, prog_bar=True)


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    
def test_segmentation_disc(img, seg_encoder, seg_model):
    img = img_transforms(img)
    
    
    if seg_encoder == 'resnet50' and seg_model == 'Unet':
        load_model = './models/segmentacao/model_disc_Unet_ResNet50.pth'
    elif seg_encoder == 'resnet50' and seg_model == 'FPN':
        load_model = './models/segmentacao/model_disc_FPN_ResNet50.pth'
    

    trainer = pl.Trainer()

    modelo = GlaucomaModel(seg_model, seg_encoder, in_channels=3, lr=0.001, loss_name='Dice', beta=None, out_classes=1)
    modelo.load_state_dict(torch.load(load_model))

    test_dataloader = DataLoader(img, batch_size=1, shuffle=True)

    with torch.no_grad():
        modelo.eval()
        logits = modelo(img)
    pr_masks = logits.sigmoid()
    mask_final = np.round(pr_masks.numpy().squeeze()).astype('uint8')
    cnts_disc = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_disc = imutils.grab_contours(cnts_disc)
    return cnts_disc






def test_segmentation_cup(img, seg_encoder, seg_model):
    img = img_transforms(img)
    if seg_encoder == 'resnet50' and seg_model == 'Unet':
        load_model = './models/segmentacao/model_cup_Unet_ResNet50.pth'
    
    elif seg_encoder == 'resnet50' and seg_model == 'FPN':
        load_model = './models/segmentacao/model_cup_FPN_ResNet50.pth'
    
    trainer = pl.Trainer()

    modelo = GlaucomaModel(seg_model, seg_encoder, in_channels=3, lr=0.001, loss_name='Dice', beta=None, out_classes=1)
    modelo.load_state_dict(torch.load(load_model))

    test_dataloader = DataLoader(img, batch_size=1, shuffle=True)

    with torch.no_grad():
        modelo.eval()
        logits = modelo(img)
    pr_masks = logits.sigmoid()
    mask_final = np.round(pr_masks.numpy().squeeze()).astype('uint8')
    cnts_cup = cv2.findContours(mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_cup = imutils.grab_contours(cnts_cup)
    return cnts_cup





