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

def profile_model(model, num_frames, num_points):
    import numpy as np
    dummy_rgbs = torch.randn(1, num_frames, 3, 256, 256).to(device)
    dummy_trajs_e = torch.randn(1, num_frames, num_points, 2).to(device)

    # 1. Warm-up (Still required!)
    # Run enough times to ensure the GPU is in a steady state
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_trajs_e, dummy_rgbs)

    # 2. Measure Multiple Iterations
    iterations = 100
    timings = []

    print(f"Profiling over {iterations} runs...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            _ = model(dummy_trajs_e, dummy_rgbs)
            end_event.record()
            
            # We must synchronize inside the loop to get accurate per-pass timing,
            # or record all events and synchronize once at the end (faster, less overhead).
            # For strict per-inference latency profiling, syncing here is acceptable
            # but adds a tiny bit of CPU overhead. 
            torch.cuda.synchronize()
            
            timings.append(start_event.elapsed_time(end_event))

    # 3. Calculate Statistics
    timings = np.array(timings)
    mean_time = np.mean(timings)
    median_time = np.median(timings)
    std_dev = np.std(timings)

    print(f"Mean latency:   {mean_time:.3f} ms")
    print(f"Median latency: {median_time:.3f} ms") # <--- Most robust metric
    print(f"Std Deviation:  {std_dev:.3f} ms")

def main(model, num_frames, num_points):
    dummy_rgbs = torch.randn(1, num_frames, 3, 256, 256).to(device)
    dummy_trajs_e = torch.randn(1, num_frames, num_points, 2).to(device)
    with torch.no_grad():
        preds, _, _, _ = model(dummy_trajs_e, dummy_rgbs)
    
    print("preds", preds[-1].shape)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Network debug')
    parser.add_argument('--profile', action='store_true', help='Model profiling')
    parser.add_argument('--num-frames', type=int, default=10, help='Number of frames to track')
    parser.add_argument('--num-points', type=int, default=64, help='Number of points to track')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    configs = {}
    model = Pips(stride=8).cuda()
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model number of parameters: {total_params}")
    if args.profile:
        profile_model(model, args.num_frames, args.num_points)
    else:
        main(model, args.num_frames, args.num_points)