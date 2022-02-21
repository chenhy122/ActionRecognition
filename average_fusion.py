
import pickle
import numpy as np
import torch
from utils import *
from train import accuracy

def total_key(dict):
    result = {}
    for key in dict.keys():
        num = len(key)
        for i in range(num):
            result[key[i]] = dict[key][i]
    return result

if __name__ == '__main__':

    rgb_pred = 'record/rgb_pred.pickle'
    cmu_pred = 'record/cmu_pred.pickle'
    lab_target = 'record/lab_target.pickle'

    with open(rgb_pred, 'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(cmu_pred, 'rb') as f:
        cmu =pickle.load(f)
    f.close()
    with open(lab_target, 'rb') as f:
        target =pickle.load(f)
    f.close()
    total = 0
    for key in rgb.keys():
        total += len(key)
    
    video_level_preds = np.zeros((total, 50))
    video_level_labels = np.zeros(total)

    rgb_total = total_key(rgb)
    cmu_total = total_key(cmu)
    target_total = total_key(target)
    ii=0
    for name in sorted(rgb_total.keys()):   
        rgb_r = rgb_total[name]
        cmu_r = cmu_total[name]
        label = int(target_total[name])
        video_level_preds[ii, :] = (rgb_r + cmu_r)
        video_level_labels[ii] = label
        ii += 1


    video_level_labels = torch.from_numpy(video_level_labels).long()
    video_level_preds = torch.from_numpy(video_level_preds).float()
        
    top1, top5 = accuracy(video_level_preds, video_level_labels, topk=(1, 5))
    
    print('   Top1 accuracy: {prec: .2f} %'.format(prec=top1.item()))
    print('   Top5 accuracy: {prec: .2f} %'.format(prec=top5.item()))
    with open('2stream_result.txt','w') as f:
        f.write(f'model1: rgb \nmodel2: cmu \ntop1: {top1.item()} \ntop5: {top5.item()}')
        
        
