import torch


class AttentionTracker:
    def __init__(self):
        # Dictionary to store attention maps, keyed by layer name
        self.attention_maps = {}

    def add(self, layer_name: str, attn_map: torch.Tensor):
        """Store an attention map for the given layer. Detach and move to CPU."""
        if layer_name not in self.attention_maps:
            self.attention_maps[layer_name] = []
        self.attention_maps[layer_name].append(attn_map.detach().cpu())

    def aggregate(self, layer_name: str):
        """Aggregate stored attention maps for a layer using elementwise average."""
        maps = self.attention_maps.get(layer_name, [])
        if not maps:
            return None
        return torch.mean(torch.stack(maps), dim=0)

    def clear(self):
        """Clear all stored attention maps."""
        self.attention_maps = {}


# Global instance that can be imported elsewhere
attention_tracker = AttentionTracker() 