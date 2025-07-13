import torch, cv2, math, os
import torch.nn.functional as F
import numpy as np
from utils.vit import vit_base_im96_patch8
from reg_wrapper import reg_model

def batch_char_recog(model, device, char_dict, char_ims, bs):
    imgs_crop   = []
    pred_output = []
    pred_prob   = []
    


    for idx, im in enumerate(char_ims):
        # 长边缩放到96, 短边等比例缩放
        h, w = im.shape[:2]
        if h > w:
            h, w = 96, round(w*96/h)
            im = cv2.resize(im, (w, h))
        else:   
            h, w = round(h*96/w), 96
            im = cv2.resize(im, (w, h))
        # 填充灰边到96*96
        x, y = round((96-w)/2), round((96-h)/2)
        crop = cv2.copyMakeBorder(im, y, 96-h-y, x, 96-w-x, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        imgs_crop.append(crop)
        
    imgs_crop = np.array(imgs_crop)

    # normalize
    imgs_crop = imgs_crop/255
    num_batches = math.ceil(len(imgs_crop)/bs)
    for idx in range(num_batches):
        if (idx+1)*bs<=imgs_crop.shape[0]:
            batch_input = torch.tensor(imgs_crop[idx*bs:(idx+1)*bs][:,np.newaxis,:,:]).type(torch.FloatTensor).to(device)
        else:
            batch_input = torch.tensor(imgs_crop[idx*bs:][:,np.newaxis,:,:]).type(torch.FloatTensor).to(device)

        output = model(batch_input)
        sm_output = F.softmax(output, dim=1)
        # 前十个字符的概率
        values, indices = torch.topk(sm_output, 10, dim=1)
        for val, idx in zip(values, indices):
            pred_output.append([char_dict[val.item()] for val in idx])
            pred_prob.append(val.cpu().detach().numpy().tolist())

    return pred_output, pred_prob


def batch_char_recog_exe(device, char_dict, char_ims, bs):
    imgs_crop   = []
    pred_output = []
    pred_prob   = []
    

    model = reg_model('./dist/reg_model')
    model.start()
    for idx, im in enumerate(char_ims):
        # 长边缩放到96, 短边等比例缩放
        h, w = im.shape[:2]
        if h > w:
            h, w = 96, round(w*96/h)
            im = cv2.resize(im, (w, h))
        else:   
            h, w = round(h*96/w), 96
            im = cv2.resize(im, (w, h))
        # 填充灰边到96*96
        x, y = round((96-w)/2), round((96-h)/2)
        crop = cv2.copyMakeBorder(im, y, 96-h-y, x, 96-w-x, cv2.BORDER_CONSTANT, value=(127, 127, 127))
        imgs_crop.append(crop)
        
    imgs_crop = np.array(imgs_crop)

    # normalize
    imgs_crop = imgs_crop/255
    num_batches = math.ceil(len(imgs_crop)/bs)
    for idx in range(num_batches):
        if (idx+1)*bs<=imgs_crop.shape[0]:
            batch_input = torch.tensor(imgs_crop[idx*bs:(idx+1)*bs][:,np.newaxis,:,:]).type(torch.FloatTensor).to(device)
        else:
            batch_input = torch.tensor(imgs_crop[idx*bs:][:,np.newaxis,:,:]).type(torch.FloatTensor).to(device)

        output = model(batch_input,device)
        sm_output = F.softmax(output, dim=1)
        # 前十个字符的概率
        values, indices = torch.topk(sm_output, 10, dim=1)
        for val, idx in zip(values, indices):
            pred_output.append([char_dict[val.item()] for val in idx])
            pred_prob.append(val.cpu().detach().numpy().tolist())
    model.cleanup()
    return pred_output, pred_prob