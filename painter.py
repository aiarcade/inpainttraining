import torch
import yaml
from saicinpainting.training.trainers.default import DefaultInpaintingTrainingModule
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import glob
import os
import tqdm
import cv2
import PIL.Image as Image
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
import torch.nn.functional as F

import io


class Painter():
    def __init__(self,config_file="config.yaml",model_file="best.ckpt",exec_device="cpu"):
        with open(config_file, 'r') as f:
            config = OmegaConf.create(yaml.safe_load(f))
            config.training_model.predict_only = True
            config.visualizer.kind = 'noop'
            kind = config.training_model.kind
            kwargs = dict(config.training_model)
            kwargs.pop('kind')
            kwargs['use_ddp'] = config.trainer.kwargs.get('accelerator', None) == 'ddp'
            self.model=DefaultInpaintingTrainingModule(config, **kwargs)
            self.device = torch.device(exec_device)
            state=torch.load(model_file,map_location=exec_device)
            self.model.load_state_dict(state['state_dict'], strict=False)
            self.model.on_load_checkpoint(state)
            self.model.freeze()
            self.model.to(self.device)

    def prepare_input_data(self,image,mask):
        input_data = dict(image=image, mask=mask[None, ...])
        input_data['image'] = self.pad_img_to_modulo(input_data['image'], 8)
        input_data['mask'] = self.pad_img_to_modulo(input_data['mask'], 8)
        return input_data


    def move_to_device(self,obj, device):
        if isinstance(obj, nn.Module):
            return obj.to(device)
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (tuple, list)):
            return [self.move_to_device(el, device) for el in obj]
        if isinstance(obj, dict):
            return {name: self.move_to_device(val, device) for name, val in obj.items()}
        raise ValueError(f'Unexpected type {type(obj)}')

    def load_image_from_file(self,fname, mode='RGB', return_orig=False):
        img = np.array(Image.open(fname).convert(mode))
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        if return_orig:
            return out_img, img
        else:
            return out_img
    def load_image_from_stream(self,stream, mode='RGB', return_orig=False):
        img = np.array(Image.open(io.BytesIO(stream)).convert(mode))
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        out_img = img.astype('float32') / 255
        if return_orig:
            return out_img, img
        else:
            return out_img


    def ceil_modulo(self,x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod


    def pad_img_to_modulo(self,img, mod):
        
        channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


    def pad_tensor_to_modulo(self,img, mod):
        batch_size, channels, height, width = img.shape
        out_height = self.ceil_modulo(height, mod)
        out_width = self.ceil_modulo(width, mod)
        return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


    def scale_image(self,img, factor, interpolation=cv2.INTER_AREA):
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = np.transpose(img, (1, 2, 0))

        img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

        if img.ndim == 2:
            img = img[None, ...]
        else:
            img = np.transpose(img, (2, 0, 1))
        return img
    def paint(self,input_data):
        with torch.no_grad():
            batch = self.move_to_device(default_collate([input_data]), self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = self.model(batch)
            cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        return cur_res

    def test_from_file(self,image_file,mask_file,out_file):
        image=self.load_image_from_file(image_file,mode='RGB')
        mask=self.load_image_from_file(mask_file,mode='L')
        data=self.prepare_input_data(image,mask)
        out_image=self.paint(data)
        cv2.imwrite(out_file, out_image)

    def predict(self,batch_dict):
        resize_to=(256, 256)
        result_images=[]
        for batch in batch_dict:
            image=self.load_image_from_stream(batch['image'],mode='RGB')
            mask=self.load_image_from_stream(batch['mask'],mode='L')
            data=self.prepare_input_data(image,mask)
            out=self.paint(data)
            out_img = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(out_img)
            result_images.append(im_pil)
        return result_images




def test_on_file():
    painter=Painter()
    image_file="samples/bertrand-gabioud-CpuFzIsHYJ0.png"
    mask_file="samples/bertrand-gabioud-CpuFzIsHYJ0_mask.png"
    painter.test_from_file(image_file,mask_file,"out.jpg")

#test_on_file()


