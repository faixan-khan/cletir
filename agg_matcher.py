import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

class CustomMultiheadAttention(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, batch_first, **kwargs):
        super().__init__(embed_dim, num_heads, batch_first=batch_first, **kwargs)
        
        # Tie the projection matrices of query and key
        self.in_proj_weight.data[embed_dim:2*embed_dim, :] = self.in_proj_weight.data[:embed_dim, :]

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        
        return super().forward(query, key, value, key_padding_mask, need_weights, attn_mask)
   
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LayerAttention(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):

        x = src
        attns = []
        for i, layer in enumerate(self.layers):
            output, attn = layer(x, x, x)
            x = torch.cat((output[:, 0, :].unsqueeze(1), x[:, 1:, :]), dim=1)
            attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns

    
class MatchATN(nn.Module):
    def __init__(self, d_model, nhead, dropout, num_encoder_layers):
        super(MatchATN, self).__init__()
        assert (d_model % 2 == 0)
        self.attn_layer = CustomMultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.self_attn = LayerAttention(self.attn_layer, num_encoder_layers)

        self.lamda = nn.Parameter(torch.tensor([0.0]))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.apply(_init_weights)

    def initialize_cls_token(self):
        target_norm = torch.empty(1).uniform_(0.01, 0.09).item()
        current_norm = torch.norm(self.cls_token.data).item()
        scaling_factor = target_norm / current_norm
        self.cls_token.data *= scaling_factor
    
    def forward(self, src_img):

        cls_embed  = torch.mean(src_img, dim=1, keepdim=True)
        
        x = torch.cat([cls_embed, src_img], 1)

        x, attn = self.self_attn(x)
        agg_embed = x[:, 0, :] 
        agg_embed = agg_embed / agg_embed.norm(dim=-1, keepdim=True)
        lamda = self.sigmoid(self.lamda)
        return agg_embed, lamda
    