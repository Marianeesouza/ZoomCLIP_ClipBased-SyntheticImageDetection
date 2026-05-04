import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop
import os

INPUT_SIZE = 224
GRID_SIZE = 16 

def is_high_info(crop, variance_threshold=18.0):
    """
    Verifica se o recorte tem informação visual (detalhes/textura) suficiente.
    """
    if crop is None or crop.size == 0:
        return False
    
    # Converter para cinza para medir o desvio padrão da textura
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, std_dev = cv2.meanStdDev(gray)
    
    # Se a variância for muito baixa, é uma área uniforme (parede, céu, etc)
    return std_dev[0][0] > variance_threshold

def process_attention_and_crop(original_image_path, attention_maps, output_dir="./attention_crops", layer_idx=2, max_crops=3):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    
    try:
        pil_img = Image.open(original_image_path).convert('RGB')
        preprocess = Compose([Resize(INPUT_SIZE, Image.BICUBIC), CenterCrop(INPUT_SIZE)])
        cropped_base_img = cv2.cvtColor(np.array(preprocess(pil_img)), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return None

    # 1. Seleção da Camada e Normalização de Dimensões
    feat = attention_maps[layer_idx].to(torch.float32)
    if feat.dim() == 3:
        if feat.shape[0] == 1: feat = feat[0]
        elif feat.shape[1] == 1: feat = feat[:, 0, :]

    num_tokens = feat.shape[0]
    if num_tokens == 257: patches = feat[1:, :]
    elif num_tokens == 256: patches = feat
    else: return None

    # 2. Cálculo de Energia
    energy = torch.norm(patches, dim=-1)
    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

    # 3. Gerar Máscara
    grid = energy_norm.reshape(GRID_SIZE, GRID_SIZE).detach().numpy()
    grid_resized = cv2.resize(grid, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap_8u = (grid_resized * 255).astype(np.uint8)
    
    thresh_val = np.percentile(heatmap_8u, 75)
    _, mask = cv2.threshold(heatmap_8u, int(thresh_val), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    # 4. Ranqueamento de TODOS os Candidatos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300: continue # Filtro mínimo de tamanho
        
        m = np.zeros(heatmap_8u.shape, dtype="uint8")
        cv2.drawContours(m, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(heatmap_8u, mask=m)[0]
        score = area * mean_intensity
        all_candidates.append((score, cnt))

    # Ordenar por importância (score)
    all_candidates.sort(key=lambda x: x[0], reverse=True)

    # 5. Geração Seletiva dos Crops (Busca até preencher a cota)
    output_filenames = []
    heatmap_vis = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_heatmap.png"), heatmap_vis)

    for score, cnt in all_candidates:
        # Se já atingimos o número máximo de crops válidos, paramos a busca
        if len(output_filenames) >= max_crops:
            break
            
        x, y, w, h = cv2.boundingRect(cnt)
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        x1, y1 = max(x - pad_w, 0), max(y - pad_h, 0)
        x2, y2 = min(x + w + pad_w, INPUT_SIZE), min(y + h + pad_h, INPUT_SIZE)
        
        crop = cropped_base_img[y1:y2, x1:x2]
        
        # VALIDAÇÃO: Só aceita se tiver "informação" (não for uniforme)
        if is_high_info(crop, variance_threshold=18.0):
            idx = len(output_filenames)
            out_path = os.path.join(output_dir, f"{base_name}_crop_{idx}.png")
            cv2.imwrite(out_path, crop)
            output_filenames.append(out_path)
            print(f"  ✓ Candidato aprovado: Crop {idx} salvo (Score: {score:.0f})")
        else:
            print(f"  [Ignorado] Candidato com alto score mas muito uniforme (Parede/Fundo).")

    return output_filenames