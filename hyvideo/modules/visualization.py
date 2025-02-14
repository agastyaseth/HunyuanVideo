from typing import Tuple, Union
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6, cmap: str = 'jet'):
    """Overlay a heatmap onto a video frame.
    
    Args:
        frame (np.ndarray): A video frame as a (H, W, 3) image. It can be in [0, 1] or [0, 255].
        heatmap (np.ndarray): A 2D array (H, W) containing normalized heat intensities.
        alpha (float): Transparency of the heatmap overlay.
        cmap (str): Colormap to use for the heatmap.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(frame, interpolation='nearest')
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, interpolation='nearest')
    plt.axis('off')
    plt.show()


def visualize_attention_for_video(tracker, tracker_name: str, video_frames: list, grid_shape: Tuple[int, int], head: Union[str, int] = 'average', alpha: float = 0.6):
    """
    Visualize the aggregated attention maps stored in the global tracker and overlay them on video frames.
    
    Args:
        tracker: The global attention tracker instance.
        tracker_name (str): The key (layer name) under which attention maps are stored.
        video_frames (list): List of video frames as numpy arrays of shape (H, W, 3).
        grid_shape (Tuple[int, int]): Tuple specifying the spatial grid dimensions (grid_h, grid_w) that tokens are arranged in per frame.
        head (Union[str, int]): Which head's attention to use ('average' to average over all heads).
        alpha (float): Transparency value for overlaying heatmap.
    """
    attn = tracker.aggregate(tracker_name)  # aggregated attention maps, assumed shape: (b, a, s, s1)
    if attn is None:
        print(f'No attention maps stored for tracker: {tracker_name}')
        return
    # Assume batch size = 1
    attn = attn[0]  # shape becomes (a, s, s1)
    if head == 'average':
        attn_map = attn.mean(dim=0)  # shape (s, s1)
    else:
        attn_map = attn[int(head)]  # shape (s, s1)
    
    # For visualization, average over the query/token dimension (s) to get one heatmap for the keys per frame
    heatmap = attn_map.mean(dim=0)  # shape (s1,)
    
    grid_h, grid_w = grid_shape
    if heatmap.numel() != grid_h * grid_w:
        print(f'Expected {grid_h * grid_w} tokens, but got {heatmap.numel()} tokens in the attention map.')
        return
    heatmap = heatmap.reshape(grid_h, grid_w).unsqueeze(0).unsqueeze(0)  # shape (1, 1, grid_h, grid_w)
    
    # Resize the heatmap to match the dimensions of the video frame
    frame_h, frame_w, _ = video_frames[0].shape
    resized_heatmap = F.interpolate(heatmap, size=(frame_h, frame_w), mode='bilinear', align_corners=False).squeeze()
    resized_heatmap = resized_heatmap.cpu().numpy()
    
    # Normalize the heatmap to [0, 1]
    resized_heatmap = (resized_heatmap - np.min(resized_heatmap)) / (np.ptp(resized_heatmap) + 1e-8)
    
    # Overlay the heatmap onto each frame
    for frame in video_frames:
        overlay_heatmap_on_frame(frame, resized_heatmap, alpha=alpha)


# Example usage:
# from hyvideo.modules.attention_tracker import attention_tracker
# from hyvideo.modules.visualization import visualize_attention_for_video
# 
# video_frames = [...]  # List of numpy arrays representing video frames
# visualize_attention_for_video(attention_tracker, 'single_stream_self_attn', video_frames, grid_shape=(8, 8), head='average', alpha=0.6) 