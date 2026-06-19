'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .resnet_mod import ChannelLinear

dict_pretrain = {
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
}


class OpenClipLinear(nn.Module):
    def __init__(self, num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=False):
        super(OpenClipLinear, self).__init__()
        
        if len(dict_pretrain[pretrain])==2:
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=dict_pretrain[pretrain][1])
        else:
            from huggingface_hub import hf_hub_download
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=hf_hub_download(*dict_pretrain[pretrain][1:]))
        
        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim
        
        self.bb = [backbone, ]
        self.normalize = normalize
        
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(OpenClipLinear, self).to(*args, **kwargs)
        return self

    def forward_features(self, x):
        with torch.no_grad():
            self.bb[0].eval()
            features = self.bb[0].encode_image(x, normalize=self.normalize)
        return features

    def forward_with_attention(self, x):
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        visual_model = self.bb[0].visual
        hooks = []

        if hasattr(visual_model, 'transformer') and hasattr(visual_model.transformer, 'resblocks'):
            # Vamos pegar as últimas 3 camadas
            for i in range(1, 4):
                layer = visual_model.transformer.resblocks[-i]
                hooks.append(layer.register_forward_hook(hook_fn))

        try:
            with torch.no_grad():
                self.bb[0].eval()
                
                # Passo 1: Converter pixels para patches (conv1)
                x_patches = visual_model.conv1(x)  # [batch, dim, h, w]
                x_patches = x_patches.reshape(x_patches.shape[0], x_patches.shape[1], -1)  # [batch, dim, 256]
                x_patches = x_patches.permute(0, 2, 1)  # [batch, 256, dim]
                
                # Adicionar CLS token e Positional Embedding
                x_patches = torch.cat([visual_model.class_embedding.to(x.dtype) + torch.zeros(x_patches.shape[0], 1, x_patches.shape[-1], dtype=x.dtype, device=x.device), x_patches], dim=1)
                x_patches = x_patches + visual_model.positional_embedding.to(x.dtype)
                x_patches = visual_model.ln_pre(x_patches)
                
                # Passo 2: Rodar o Transformer
                x_patches = x_patches.permute(1, 0, 2)  # [257, batch, dim]
                x_features = visual_model.transformer(x_patches)
                
                # Passo 3: Finalização
                x_features = x_features.permute(1, 0, 2)  # [batch, 257, dim]
                features = visual_model.ln_post(x_features[:, 0, :]) # Pega apenas o CLS
                if visual_model.proj is not None:
                    features = features @ visual_model.proj
                    
        finally:
            for h in hooks:
                h.remove()

        # Invertemos para que a última camada seja a última da lista
        activations.reverse() 
        return features, activations

    def forward_head(self, x):
        return self.fc(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))
