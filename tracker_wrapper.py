import torch
import numpy as np
import sys
import os

# Add the directory containing this file to sys.path to allow imports
# This works both when run directly and when imported as a module
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import saverloader
from nets.pips2 import Pips
import torch.nn.functional as F
from utils.basic import print_stats

class Pips2Tracker:
    def __init__(self, configs, track_length=8):
        self.configs = configs
        self.score_threshold = -1
        self.iters = 16
        current_dir = os.path.dirname(os.path.abspath(__file__))
        init_dir = os.path.join(current_dir, 'reference_model')
        self.model = Pips(stride=8).cuda()
        if init_dir:
            print('loading model from', init_dir)
            _ = saverloader.load(init_dir, self.model)
        self.model.eval()


    def track(self, prevImg, nextImg, prevPts):
        if isinstance(nextImg, list):
            images = [prevImg] + nextImg
        else:
            images = [prevImg, nextImg]

        imgs_np = np.stack(images, axis=0)  # (T, H, W, C)
        imgs_np = imgs_np.astype(np.float32)
        imgs_np = np.transpose(imgs_np, (0, 3, 1, 2))
        rgbs = torch.from_numpy(imgs_np).unsqueeze(0)  # (1, T, C, H, W)
        rgbs = rgbs.cuda().float() # B, S, C, H, W

        B, S, C, H, W = rgbs.shape
        
        # Scale points from original image space to resized space
        xy0 = torch.from_numpy(prevPts).float().unsqueeze(0).cuda()  # (1, N, 2)

        # zero-vel init
        trajs_e = xy0.unsqueeze(1).repeat(1,S,1,1)
        
        with torch.no_grad():
            preds, _, _, _ = self.model(trajs_e, rgbs, iters=self.iters, feat_init=None, beautify=True)
    
        trajs_e = preds[-1]
        pred_pts = trajs_e.squeeze(0)[1:].detach().cpu().numpy()
        status = np.ones((pred_pts.shape[0], pred_pts.shape[1]), dtype=bool)

        return pred_pts, status

if __name__ == '__main__':
    
    configs = {}
    tracker = Pips2Tracker(configs)
    prevImg = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    nextImg = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(7)]
    N = 1000  # number of keypoints
    height, width = prevImg.shape[:2]
    prevPts = np.stack([
        np.random.randint(0, width, size=N),
        np.random.randint(0, height, size=N)
    ], axis=-1)
    pred_pts, status = tracker.track(prevImg, nextImg, prevPts)
    print('pred_pts', pred_pts.shape)
    print('status', status.shape)