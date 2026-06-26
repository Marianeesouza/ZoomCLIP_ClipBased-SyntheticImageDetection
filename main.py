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
import os
import pandas
import numpy as np
import tqdm
import glob
import sys
import yaml
from PIL import Image

from torchvision.transforms  import CenterCrop, Resize, Compose, InterpolationMode
from utils.processing import make_normalize
from utils.fusion import apply_fusion
from utils.crop_processing import process_attention_and_crop
from networks import create_architecture, load_weights


def get_config(model_name, weights_dir='./weights'):
    with open(os.path.join(weights_dir, model_name, 'config.yaml')) as fid:
        data = yaml.load(fid, Loader=yaml.FullLoader)
    model_path = os.path.join(weights_dir, model_name, data['weights_file'])
    return data['model_name'], model_path, data['arch'], data['norm_type'], data['patch_size']


def runnig_tests(input_csv, weights_dir, models_list, device, batch_size = 1, extract_attention=False, attention_output_dir=None):
    table = pandas.read_csv(input_csv)[['filename',]]
    rootdataset = os.path.dirname(os.path.abspath(input_csv))
    
    models_dict = dict()
    transform_dict = dict()
    print("Models:")
    for model_name in models_list:
        print(model_name, flush=True)
        print(f"Carregando {model_name}...", flush=True)
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        # original weights
        global_model = load_weights(create_architecture(arch), model_path)
        global_model = global_model.to(device).eval()

        # local model weights
        local_model = global_model

        if "clipdet" in model_name: 
            local_model = load_weights(create_architecture(arch), model_path)
            try:
                local_weight_path = os.path.join('weights', 'local_head', 'local_head_weights.pth')
                local_weights = torch.load(local_weight_path, map_location=device, weights_only=True)
                local_model.load_state_dict(local_weights, strict=False)
                local_model = local_model.to(device).eval()

            except Exception as e:
                print(f"Load local weights error: {e}")

        transform = list()
        if patch_size is None:
            print('input none', flush=True)
            transform_key = 'none_%s' % norm_type
        elif patch_size=='Clip224':
            print('input resize:', 'Clip224', flush=True)
            transform.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
            transform.append(CenterCrop((224, 224)))
            transform_key = 'Clip224_%s' % norm_type
        elif isinstance(patch_size, tuple) or isinstance(patch_size, list):
            print('input resize:', patch_size, flush=True)
            transform.append(Resize(*patch_size))
            transform.append(CenterCrop(patch_size[0]))
            transform_key = 'res%d_%s' % (patch_size[0], norm_type)
        elif patch_size > 0:
            print('input crop:', patch_size, flush=True)
            transform.append(CenterCrop(patch_size))
            transform_key = 'crop%d_%s' % (patch_size, norm_type)
        
        transform.append(make_normalize(norm_type))
        transform = Compose(transform)
        transform_dict[transform_key] = transform
        models_dict[model_name] = (transform_key, global_model, local_model) 
        print(flush=True)

    ### test
    with torch.no_grad():
        
        do_models = list(models_dict.keys())
        do_transforms = set([models_dict[_][0] for _ in do_models])
        print(do_models)
        print(do_transforms)
        print(flush=True)
        
        print("Running the Tests")
        batch_img = {k: list() for k in transform_dict}
        batch_id = list()
        last_index = table.index[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            for k in transform_dict:
                batch_img[k].append(transform_dict[k](Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index==last_index):
                for k in do_transforms:
                    batch_img[k] = torch.stack(batch_img[k], 0)

                for model_name in do_models:
                    global_model = models_dict[model_name][1]
                    local_model = models_dict[model_name][2]
                    
                    input_key = models_dict[model_name][0]
                    input_tensor = batch_img[input_key].clone().to(device)

                    with torch.no_grad():
                        full_output = global_model(input_tensor).cpu().numpy()
                        
                        if full_output.shape[1] == 1:
                            global_logits = full_output[:, 0]
                        elif full_output.shape[1] == 2:
                            global_logits = full_output[:, 1] - full_output[:, 0]
                        else:
                            global_logits = np.mean(full_output, (1, 2))

                    if extract_attention and hasattr(global_model, 'forward_with_attention'):
                        features, activations = global_model.forward_with_attention(input_tensor)
                        
                        final_logits_batch = []
                        for b_idx in range(len(batch_id)):
                            file_idx = batch_id[b_idx]
                            g_logit = global_logits[b_idx]
                            filename = os.path.join(rootdataset, table.loc[file_idx, 'filename'])
                            
                            img_activation = [act[:, b_idx, :] for act in activations]
                            
                            pil_crops_list = process_attention_and_crop(
                                filename, 
                                img_activation, 
                                output_dir=attention_output_dir or "./attention_crops",
                                max_crops=5,
                                save_images=False
                            )

                            img_local_logits = []

                            if pil_crops_list:
                                for crop_img in pil_crops_list:

                                    crop_tensor = transform_dict[input_key](crop_img).unsqueeze(0).to(device)
                                    
                                    with torch.no_grad():
                                        l_res = local_model(crop_tensor).cpu().numpy().flatten()
                                        img_local_logits.append(l_res[0])
                            
                            # Se nenhum crop for validado pelo filtro de variância, penaliza o score local
                            if len(img_local_logits) == 0:
                                l_logit = -10.0
                            else:
                                logits_arr = np.array(img_local_logits)
                                confidences = np.abs(logits_arr)
                                exps = np.exp(confidences - np.max(confidences))
                                weights = exps / np.sum(exps)
                                l_logit = np.sum(weights * logits_arr)
                        
                            combined_score = max(g_logit, l_logit)
                            
                            # Salvando os logs base na tabela
                            table.loc[file_idx, f'{model_name}_global'] = g_logit
                            table.loc[file_idx, f'{model_name}_local'] = l_logit
                            table.loc[file_idx, f'{model_name}_fusiongl'] = combined_score
                            final_logits_batch.append(combined_score)
                        
                        logit1 = np.array(final_logits_batch)
                    
                    else:
                        out_tens = global_model(input_tensor).cpu().numpy()
                        
                        if out_tens.shape[1] == 1:
                            logit1 = out_tens[:, 0]
                        elif out_tens.shape[1] == 2:
                            logit1 = out_tens[:, 1] - out_tens[:, 0]
                        else:
                            logit1 = np.mean(out_tens, (1, 2))

                    for b_idx, f_idx in enumerate(batch_id):
                        table.loc[f_idx, model_name] = logit1[b_idx]

                batch_img = {k: list() for k in transform_dict}
                batch_id = list()

            assert len(batch_id)==0
        
    return table


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_csv"     , '-i', type=str, help="The path of the input csv file with the list of images")
    parser.add_argument("--out_csv"    , '-o', type=str, help="The path of the output csv file", default="./results.csv")
    parser.add_argument("--weights_dir", '-w', type=str, help="The directory to the networks weights", default="./weights")
    parser.add_argument("--models"     , '-m', type=str, help="List of models to test", default='clipdet_latent10k_plus,Corvi2023')
    parser.add_argument("--fusion"     , '-f', type=str, help="Fusion function", default='soft_or_prob')
    parser.add_argument("--device"     , '-d', type=str, help="Torch device", default='cuda:0')
    parser.add_argument("--extract_attention", action='store_true', help="Extract attention maps and crop high-attention regions")
    parser.add_argument("--attention_output_dir", type=str, help="Directory to save attention crops", default="./attention_crops")
    args = vars(parser.parse_args())
    
    if args['models'] is None:
        args['models'] = os.listdir(args['weights_dir'])
    else:
        args['models'] = args['models'].split(',')
    
    table = runnig_tests(
        args['in_csv'], 
        args['weights_dir'], 
        args['models'], 
        args['device'],
        extract_attention=args['extract_attention'],
        attention_output_dir=args['attention_output_dir']
    )

    if args['extract_attention']:
        for model_name in args['models']:
            if f'{model_name}_global' in table.columns:
                g_scores = table[f'{model_name}_global'].values
                l_scores = table[f'{model_name}_local'].values

                # --- Z-SCORE NORMALIZATION ---
                g_std = np.std(g_scores) + 1e-7
                l_std = np.std(l_scores) + 1e-7
                
                g_norm = (g_scores - np.mean(g_scores)) / g_std
                l_norm = (l_scores - np.mean(l_scores)) / l_std

                # --- ESTRATÉGIAS DE INTEGRAÇÃO GLOBAL-LOCAL ---
                
                # Teste A: Soft OR Probabilístico
                p_g = 1 / (1 + np.exp(-g_norm))
                p_l = 1 / (1 + np.exp(-l_norm))
                p_fusion = 1 - (1 - p_g) * (1 - p_l)
                table[f'{model_name}_fusion_soft_or'] = np.log(p_fusion / (1 - p_fusion + 1e-7))

                # Teste B: Pesos Fixos
                table[f'{model_name}_fusion_w_70G_30L'] = (0.7 * g_norm) + (0.3 * l_norm)
                table[f'{model_name}_fusion_w_50G_50L'] = (0.5 * g_norm) + (0.5 * l_norm)
                table[f'{model_name}_fusion_w_30G_70L'] = (0.3 * g_norm) + (0.7 * l_norm)

                # Teste C: Max-Pooling
                table[f'{model_name}_fusion_max_calibrated'] = np.maximum(g_norm, l_norm)

                # Teste D: Fusão Dinâmica por Confiança (Softmax Global vs Local)
                conf_g = np.abs(g_norm)
                conf_l = np.abs(l_norm)
                
                exp_g = np.exp(conf_g)
                exp_l = np.exp(conf_l)
                sum_exp = exp_g + exp_l
                
                w_g = exp_g / sum_exp
                w_l = exp_l / sum_exp
                
                table[f'{model_name}_fusion_dynamic'] = (w_g * g_norm) + (w_l * l_norm)

    if args['fusion'] is not None:
        fusion_data = table[args['models']].copy()
        
        for col in args['models']:
            col_data = fusion_data[col].values
            col_std = np.std(col_data) + 1e-7
            fusion_data[col] = (col_data - np.mean(col_data)) / col_std
            
        table['fusion'] = apply_fusion(fusion_data.values, args['fusion'], axis=-1)
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
