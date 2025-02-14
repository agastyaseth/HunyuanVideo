import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.attention_tracker import attention_tracker
from hyvideo.modules.visualization import visualize_attention_for_video
import numpy as np
import torch


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            base_name = f"{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}"
            
            # Save video
            video_path = f"{save_path}/{base_name}.mp4"
            save_videos_grid(sample, video_path, fps=24)
            logger.info(f'Sample saved to: {video_path}')
            
            # If dump_attention flag is set, save attention maps
            if args.dump_attention:
                # Extract frames from the video for visualization
                frames = [np.array(frame) for frame in sample[0]]  # Convert tensor to numpy arrays
                
                # Calculate grid shape based on the model's configuration
                grid_h = args.video_size[0] // 16  # Assuming patch size of 16
                grid_w = args.video_size[1] // 16
                
                # Visualize attention maps for each layer that was tracked
                for layer_name in attention_tracker.attention_maps.keys():
                    logger.info(f'Visualizing attention maps for layer: {layer_name}')
                    visualize_attention_for_video(
                        attention_tracker,
                        layer_name,
                        frames,
                        grid_shape=(grid_h, grid_w),
                        head='average'
                    )
                
                # Clear the tracker after visualization
                attention_tracker.clear()

if __name__ == "__main__":
    main()
