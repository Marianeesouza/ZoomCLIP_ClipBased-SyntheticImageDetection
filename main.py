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
        _, model_path, arch, norm_type, patch_size = get_config(model_name, weights_dir=weights_dir)

        model = load_weights(create_architecture(arch), model_path)
        model = model.to(device).eval()

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
        models_dict[model_name] = (transform_key, model)
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
                    model_instance = models_dict[model_name][1]
                    input_key = models_dict[model_name][0]
                    input_tensor = batch_img[input_key].clone().to(device)

                    with torch.no_grad():
                        full_output = model_instance(input_tensor).cpu().numpy()
                        
                        if full_output.shape[1] == 1:
                            global_logits = full_output[:, 0]
                        elif full_output.shape[1] == 2:
                            global_logits = full_output[:, 1] - full_output[:, 0]
                        else:
                            global_logits = np.mean(full_output, (1, 2))

                    if extract_attention and hasattr(model_instance, 'forward_with_attention'):
                        features, activations = model_instance.forward_with_attention(input_tensor)
                        
                        final_logits_batch = []
                        for b_idx in range(len(batch_id)):
                            file_idx = batch_id[b_idx]
                            g_logit = global_logits[b_idx]
                            filename = os.path.join(rootdataset, table.loc[file_idx, 'filename'])
                            
                            # Extrai ativação específica
                            img_activation = [act[:, b_idx, :] for act in activations]
                            
                            # Gera o crop
                            crop_paths = process_attention_and_crop(
                                filename, 
                                img_activation, 
                                output_dir=attention_output_dir or "./attention_crops",
                                max_crops=5
                            )
                            
                            l_logit = -10.0 # Default caso falhe o crop
                            
                            if crop_paths:
                                crop_img = Image.open(crop_paths[0]).convert('RGB')
                                crop_tensor = transform_dict[input_key](crop_img).unsqueeze(0).to(device)
                                
                                with torch.no_grad():
                                    l_res = model_instance(crop_tensor).cpu().numpy().flatten()
                                    l_logit = l_res[0]
                            
                            combined_score = max(g_logit, l_logit)
                            
                            table.loc[file_idx, f'{model_name}_global'] = g_logit
                            table.loc[file_idx, f'{model_name}_local'] = l_logit
                            table.loc[file_idx, f'{model_name}_fusiongl'] = combined_score
                            final_logits_batch.append(combined_score)
                        
                        logit1 = np.array(final_logits_batch)
                    
                    else:
                        out_tens = model_instance(input_tensor).cpu().numpy()
                        
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
    if args['fusion'] is not None:
        table['fusion'] = apply_fusion(table[args['models']].values, args['fusion'], axis=-1)
    
    output_csv = args['out_csv']
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    table.to_csv(output_csv, index=False)  # save the results as csv file
    
