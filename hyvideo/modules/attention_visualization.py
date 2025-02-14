import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import seaborn as sns
from einops import rearrange

from .attention_tracker import attention_tracker


class AttentionVisualizer:
    """Utility class to visualize attention maps from HunyuanVideo's transformer blocks."""
    
    @staticmethod
    def get_attention_map(layer_name: str) -> Optional[torch.Tensor]:
        """Retrieve and aggregate attention map for a specific layer."""
        return attention_tracker.aggregate(layer_name)
    
    @staticmethod
    def plot_attention_heatmap(
        attn_map: torch.Tensor,
        title: str,
        save_path: Optional[str] = None,
        head_idx: Optional[int] = None,
        batch_idx: int = 0,
        cmap: str = 'viridis',
        figsize: tuple = (10, 8)
    ):
        """Plot attention heatmap for a specific head or average across heads."""
        # Move to CPU and convert to numpy
        attn_map = attn_map.detach().cpu()
        
        # Handle batch dimension
        if len(attn_map.shape) == 4:  # [batch, heads, seq_len_q, seq_len_k]
            attn_map = attn_map[batch_idx]
            
        if head_idx is not None:
            # Plot specific attention head
            attn_matrix = attn_map[head_idx].numpy()
            title = f"{title} (Head {head_idx})"
        else:
            # Average across heads
            attn_matrix = attn_map.mean(dim=0).numpy()
            title = f"{title} (Averaged across heads)"
            
        plt.figure(figsize=figsize)
        sns.heatmap(attn_matrix, cmap=cmap)
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def visualize_cross_attention(
        block_idx: int,
        save_dir: Optional[str] = None,
        head_idx: Optional[int] = None,
        batch_idx: int = 0
    ):
        """Visualize cross-attention maps for a specific double block."""
        # Get img->txt and txt->img cross attention maps
        img_txt_map = AttentionVisualizer.get_attention_map(f"double_block_{block_idx}_img_txt_cross_attn")
        txt_img_map = AttentionVisualizer.get_attention_map(f"double_block_{block_idx}_txt_img_cross_attn")
        
        if img_txt_map is not None:
            AttentionVisualizer.plot_attention_heatmap(
                img_txt_map,
                f"Image->Text Cross Attention (Block {block_idx})",
                save_path=f"{save_dir}/block_{block_idx}_img_txt_cross_attn.png" if save_dir else None,
                head_idx=head_idx,
                batch_idx=batch_idx
            )
            
        if txt_img_map is not None:
            AttentionVisualizer.plot_attention_heatmap(
                txt_img_map,
                f"Text->Image Cross Attention (Block {block_idx})",
                save_path=f"{save_dir}/block_{block_idx}_txt_img_cross_attn.png" if save_dir else None,
                head_idx=head_idx,
                batch_idx=batch_idx
            )
    
    @staticmethod
    def visualize_self_attention(
        block_idx: int,
        block_type: str = "double",
        save_dir: Optional[str] = None,
        head_idx: Optional[int] = None,
        batch_idx: int = 0
    ):
        """Visualize self-attention maps for a specific block."""
        if block_type == "double":
            # Get image and text self-attention maps
            img_map = AttentionVisualizer.get_attention_map(f"double_block_{block_idx}_img_self_attn")
            txt_map = AttentionVisualizer.get_attention_map(f"double_block_{block_idx}_txt_self_attn")
            
            if img_map is not None:
                AttentionVisualizer.plot_attention_heatmap(
                    img_map,
                    f"Image Self Attention (Block {block_idx})",
                    save_path=f"{save_dir}/block_{block_idx}_img_self_attn.png" if save_dir else None,
                    head_idx=head_idx,
                    batch_idx=batch_idx
                )
                
            if txt_map is not None:
                AttentionVisualizer.plot_attention_heatmap(
                    txt_map,
                    f"Text Self Attention (Block {block_idx})",
                    save_path=f"{save_dir}/block_{block_idx}_txt_self_attn.png" if save_dir else None,
                    head_idx=head_idx,
                    batch_idx=batch_idx
                )
        else:
            # Get unified attention map from single stream block
            unified_map = AttentionVisualizer.get_attention_map(f"single_block_{block_idx}_unified_attn")
            if unified_map is not None:
                AttentionVisualizer.plot_attention_heatmap(
                    unified_map,
                    f"Unified Attention (Block {block_idx})",
                    save_path=f"{save_dir}/block_{block_idx}_unified_attn.png" if save_dir else None,
                    head_idx=head_idx,
                    batch_idx=batch_idx
                )
    
    @staticmethod
    def visualize_attention_flow(
        block_indices: List[int],
        attention_type: str,
        save_dir: Optional[str] = None,
        head_idx: Optional[int] = None,
        batch_idx: int = 0,
        num_cols: int = 4
    ):
        """Visualize how attention patterns evolve across layers."""
        num_blocks = len(block_indices)
        num_rows = (num_blocks + num_cols - 1) // num_cols
        
        plt.figure(figsize=(5*num_cols, 5*num_rows))
        
        for idx, block_idx in enumerate(block_indices):
            plt.subplot(num_rows, num_cols, idx + 1)
            
            attn_map = AttentionVisualizer.get_attention_map(f"{attention_type}_{block_idx}")
            if attn_map is not None:
                attn_map = attn_map.detach().cpu()
                if head_idx is not None:
                    attn_matrix = attn_map[batch_idx, head_idx].numpy()
                else:
                    attn_matrix = attn_map[batch_idx].mean(dim=0).numpy()
                    
                sns.heatmap(attn_matrix, cmap='viridis')
                plt.title(f"Block {block_idx}")
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/attention_flow_{attention_type}.png")
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def get_most_attended_positions(
        attn_map: torch.Tensor,
        head_idx: Optional[int] = None,
        batch_idx: int = 0,
        top_k: int = 5
    ) -> List[tuple]:
        """Get the positions with highest attention weights."""
        attn_map = attn_map.detach().cpu()
        
        if head_idx is not None:
            attn_matrix = attn_map[batch_idx, head_idx]
        else:
            attn_matrix = attn_map[batch_idx].mean(dim=0)
            
        # Find top-k attention weights
        flat_indices = torch.topk(attn_matrix.view(-1), k=top_k).indices
        rows = flat_indices // attn_matrix.shape[1]
        cols = flat_indices % attn_matrix.shape[1]
        
        positions = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            weight = attn_matrix[r, c].item()
            positions.append((r, c, weight))
            
        return positions 