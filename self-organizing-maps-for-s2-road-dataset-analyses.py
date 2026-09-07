#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:08:17 2025

@author: usuario
"""

import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import time, os
from datetime import datetime
import tifffile as tiff
import rasterio
import shutil
from utils.functions import crop_and_save_image, skeletonize_mask, S2_RI, S2_NDBI, S2_NDVI, S2_NDRI, S2_NDWI, prepare_dataset_vs2, PrioriProb, PostProb, CheckSamples_vs1

############################################## PARAMETROS ###################################################
#Set the bands to be processed
band_list = ['b2', 'b3', 'b4', 'b8', 'b11', 'b12']

#set the indexes to be processed
index_list = []#['ndbi', 'ndvi', 'ndwi', 'ri', 'ndri']

#merges the index_list and band_list groups
channel_list = band_list + index_list
print('Channel List:', channel_list)

IMG_CHANNELS = len(channel_list)

#define o novo tamanho das imagens #img.shape = (Height, Width)
NS = [512, 512]

#=================================================================

#Parametros de filtragem de Mascaras
save_masks = True
WindowSize = [1,1]
best_qe = 0.0
    
#proporcao de positivos/negativos ('0': para manter a proporção dos dados; ou outro valor para especificar a proporção de positivos e negativos)
balance_p = 0

#numero de imagens do dataset a serem escolhidas para as amostras
nr_images = 60

# Define N_total como o número total de amostras desejadas do dataset
N_total = 1000000
test_sample = 10000

# ---------- Rede SOM ----------
MS = [20, 20]  # Tamanho do mapa SOM
epochs = 200           # Número de ciclos de ajuste fino
batch_size = 400       # Tamanho de cada atualização incremental
sigma_f = 1.5
l_r = 0.1
ngbh_function = 'gaussian'
topology = 'rectangular'
# Define um total de amostras desejadas para o treinamento da rede
N_pts = 50000
# -------------------------------

######################################### PREPARO DOS DIRETÓRIOS ############################################
PROJECT_DIR = Path(__file__).resolve().parent

DS_DIR = PROJECT_DIR.parent / "Dataset" / "Test"

#Nome da mascara de referência
ref_mask = 'Mask'

# Define os endereços e cria folders temporários
Temp_Folder = PROJECT_DIR / "Results"
if not os.path.exists(Temp_Folder):
    os.makedirs(Temp_Folder, exist_ok=True)
    
if save_masks:
    masks_filtered_Folder = os.path.join(Temp_Folder, f'Testes_{WindowSize[0]}x{WindowSize[1]}',f'Masks_Filtered_{WindowSize[0]}x{WindowSize[1]}')
    if not os.path.exists(masks_filtered_Folder):
        os.makedirs(masks_filtered_Folder, exist_ok=True)

#Cria um folder para o teste
Test_Folder = os.path.join(Temp_Folder, f"SOM_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(Test_Folder, exist_ok=True)

#Folder de outputs
OutputFolder = os.path.join(Test_Folder, 'Outputs')
if not os.path.exists(OutputFolder):
    os.makedirs(OutputFolder, exist_ok=True)
    
##################################### Verifica as imagens ######################################################
#Check the images
'''checks that all images comply with the dimensions specified in the parameters and resizes them if necessary '''

#Temp folder for the processed imgs
Temp_Imgs_Folder = os.path.join(Temp_Folder, 'Temp_Imgs')
if not os.path.exists(Temp_Imgs_Folder):
    os.makedirs(Temp_Imgs_Folder, exist_ok=True)

#creates a dictionary to store the addresses of the bands and masks
ch_url_dict = {band: i for i, band in enumerate(channel_list)}
ch_url_dict['Mask'] = 0

# Get the list of images
image_files = os.listdir(os.path.join(DS_DIR, 'Imgs', channel_list[0]))

# Filter only image files, if necessary (.tif)
image_files = [f for f in image_files if f.lower().endswith(('.tif', '.tiff'))]

#Filtra as bandas disponiveis no diretorio do dataset
# Lista todos os itens dentro de DS_DIR
itens = os.listdir(DS_DIR)
# Filtra apenas os diretórios (pastas)
pastas = [item for item in itens if os.path.isdir(os.path.join(DS_DIR, item))]

#checks the size of masks
out_of_size = 0
for img in tqdm(image_files, desc="Checking the mask sizes..."):
#for i, img in enumerate(image_files):
    img_url = os.path.join(DS_DIR, ref_mask, img)
    b_img = np.array(tiff.imread(img_url))
    
    if b_img.shape != tuple(NS):
        out_of_size = 1
        #stop the image size verification loop
        
        print('\n Encontrada uma mascara fora do tamanho previsto, realizando o recorte das mascaras...\n')        
  
        #input folder
        input_folder = os.path.join(DS_DIR, ref_mask)
    
        #output folder
        masks_folder = os.path.join(Temp_Imgs_Folder, 'Mask')
        if not os.path.exists(masks_folder):
            os.makedirs(masks_folder, exist_ok=True)
    
        #update the url for the channel
        ch_url_dict['Mask'] = masks_folder
        
        #resizes all the images in the channel
        for img_name in tqdm(image_files, desc=" - Redimensionando mascaras"):
            #open image
            b_img = np.array(tiff.imread(os.path.join(input_folder, img_name)))
             
            #resize the image
            resized_img = crop_and_save_image(b_img, input_folder, masks_folder, NS, is_mask = True)
    else:
        #update the url for the channel
        ch_url_dict['Mask'] = os.path.join(DS_DIR, ref_mask)
    
#=============================================================

# Esqueletização de mascaras
for img in tqdm(image_files, desc=" - Esqueletizando mascaras"):
    
    img_url = os.path.join(DS_DIR, ref_mask, img)
    
    # ---- Leitura preservando metadados ----
    with rasterio.open(img_url) as src:
        b_img = src.read(1)  # Lê primeira banda
        profile = src.profile.copy()  # Copia metadados
        transform = src.transform
        crs = src.crs
    
    # Binariza
    b_img = (b_img > 0).astype(np.uint8)
    
    # Gera o esqueleto
    sk_mask = skeletonize_mask(b_img, im_show=False)
    sk_mask = sk_mask.astype(np.uint8)

    # Pasta de saída
    out_folder = os.path.join(Temp_Imgs_Folder, 'Mask')
    os.makedirs(out_folder, exist_ok=True)

    out_path = os.path.join(out_folder, img)

    # ---- Atualiza perfil para escrita ----
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'  # opcional, mas recomendado
    )

    # ---- Escrita preservando orientação espacial ----
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(sk_mask, 1)

# Atualiza o caminho do canal
ch_url_dict['Mask'] = out_folder

#=============================================================
        
#checks the channels
for ch in tqdm(channel_list, desc="Checking the channels"):
    #checks if the channel is available in the dataset
    #print(f'Checking the channel {ch}...\n')
    if os.path.isdir(os.path.join(DS_DIR, 'Imgs', ch)):
        #checks the size of images
        out_of_size = 0
        for i, img in enumerate(image_files):
            img_url = os.path.join(DS_DIR, 'Imgs', ch, img)
            b_img = np.array(tiff.imread(img_url))
            if b_img.shape != tuple(NS):
                out_of_size = 1
                #stop the image size verification loop
                break
        if out_of_size == 1:   
            #input folder
            input_folder = os.path.join(DS_DIR, 'Imgs', ch)

            #output folder
            out_folder = os.path.join(Temp_Imgs_Folder, ch)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder, exist_ok=True)

            #update the url for the channel
            ch_url_dict[ch] = out_folder
            
            #resizes all the images in the channel
            print(f'- Images out of size, processing the images of channel {ch}')
            for img_name in tqdm(image_files, desc=f" - Resizing the channel {ch}"):
                #open image
                b_img = np.array(tiff.imread(os.path.join(input_folder, img_name)))
                #resize the image
                resized_img = crop_and_save_image(b_img, input_folder, out_folder, NS)
        else:
            #update the url for the channel
            ch_url_dict[ch] = os.path.join(DS_DIR, 'Imgs', ch)
    else:
        #processes a non-existent index or band
        ###########################################################################
        #verify the band folder
        out_folder = os.path.join(Temp_Imgs_Folder, ch)
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder, exist_ok=True)

        #update the url for the channel
        ch_url_dict[ch] = out_folder
            
        if ch == 'ri':
            for idx, img_name in enumerate(image_files):
                banda_2_path = os.path.join(DS_DIR, 'Imgs', 'b2', img_name)
                banda_8_path = os.path.join(DS_DIR, 'Imgs', 'b8', img_name)
                banda_11_path = os.path.join(DS_DIR, 'Imgs', 'b11', img_name) 
                ri_img_path = os.path.join(out_folder, img_name)
                
                #calcular o indice RI
                try:
                    S2_RI(banda_2_path, banda_8_path, banda_11_path, ri_img_path)
                except:
                    print(f'- Nao foi possivel calcular o indice {ch}')
            
            ri_img = idx + 1
            print(f"- Road index processed with {ri_img} samples\n")
    
        if ch == 'ndri':
            for idx, img_name in enumerate(image_files):
                banda_2_path = os.path.join(DS_DIR, 'Imgs', 'b2', img_name)
                banda_8_path = os.path.join(DS_DIR, 'Imgs', 'b8', img_name)
                ndri_img_path = os.path.join(out_folder, img_name)

                try:
                    #calcular o indice RI
                    S2_NDRI(banda_2_path, banda_8_path, ndri_img_path)
                except:
                    print(f'- Nao foi possivel calcular o indice {ch}')
            ri_img = idx + 1
            print(f"- Road index processed with {ri_img} samples\n")
    
        if ch == 'ndbi':
            for idx, img_name in enumerate(image_files):
                banda_8_path = os.path.join(DS_DIR, 'Imgs', 'b8', img_name)
                banda_11_path = os.path.join(DS_DIR, 'Imgs', 'b11', img_name) 
                ndbi_img_path = os.path.join(out_folder, img_name)
                
                try:
                    #calcular o indice NDBI
                    S2_NDBI(banda_8_path, banda_11_path, ndbi_img_path)
                except:
                    print(f'- Nao foi possivel calcular o indice {ch}')
                    
            ndbi_img = idx + 1
            print(f"- NDBI index processed with {ndbi_img} samples\n")
    
        if ch == 'ndvi':
            for idx, img_name in enumerate(image_files):
                banda_4_path = os.path.join(DS_DIR, 'Imgs', 'b4', img_name)
                banda_8_path = os.path.join(DS_DIR, 'Imgs', 'b8', img_name)
                ndvi_img_path = os.path.join(out_folder, img_name)
                
                try:
                    #calcular o indice NDVI
                    S2_NDVI(banda_8_path, banda_4_path, ndvi_img_path)
                except:
                    print(f'- Nao foi possivel calcular o indice {ch}')
            
            ndvi_img = idx + 1
            print(f"- NDVI index processed with {ndvi_img} samples\n")
            
        if ch == 'ndwi':
            for idx, img_name in enumerate(image_files):
                banda_3_path = os.path.join(DS_DIR, 'Imgs', 'b3', img_name)
                banda_8_path = os.path.join(DS_DIR, 'Imgs', 'b8', img_name)
                ndwi_img_path = os.path.join(out_folder, img_name)

                try:
                    #calcular o indice NDWI
                    S2_NDWI(banda_8_path, banda_3_path, ndwi_img_path)
                except:
                    print(f'- Nao foi possivel calcular o indice {ch}')
                    
            ndwi_img = idx + 1
            print(f"- NDWI index processed with {ndwi_img} samples\n")
    
##################################
print('Diretorios das bandas:')
band_ref = 0
for b in ch_url_dict:
    if band_ref == 0:
        band_ref = [b, ch_url_dict[b]]
    print(f'Banda: {b} : {ch_url_dict[b]}')

#verifica os arquivos salvos nas pastas temporarias
Imgs_List = os.listdir(band_ref[1])
print(str(len(Imgs_List)) + ' arquivos de treinamento verificados com sucesso.')

Labels_List = os.listdir(ch_url_dict['Mask'])
print(str(len(Labels_List)) + ' mascaras de treinamento no folder: ' + str(ch_url_dict['Mask']) + '\n')

#################################################################################################
 
# ============================================================
# LIMPEZA DE SAÍDAS ANTERIORES
# ============================================================

arquivos = os.listdir(OutputFolder)

if len(arquivos) > 0:
    ask_del = input(f'Encontrados {len(arquivos)} arquivos na pasta {OutputFolder}. Deseja apagar os arquivos anteriores (y/n)? ')
    if ask_del.lower() == 'y':
        for filename in arquivos:
            file_path = os.path.join(OutputFolder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Erro ao apagar {file_path}: {e}")
        print("Arquivos anteriores removidos.")
    else:
        print("Mantendo arquivos anteriores.")

#pasta de saída de mascaras
if save_masks:
    #Temp folder for the processed imgs
    #masks_filtered_Folder = os.path.join(OutputFolder, f"Masks_Filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    #os.makedirs(masks_filtered_Folder, exist_ok=True)
    processed_imgs = os.listdir(masks_filtered_Folder)
    print(f'Encontradas {len(processed_imgs)} mascaras processadas.')
else:
    processed_imgs = []

#Calcula a diferenca entre a lista de imagens e o cjt de imagens processadas:
remaining_imgs = [img for img in Imgs_List if img not in processed_imgs]

#reinicia a lista de imgs processadas
processed_imgs = []

print('\nIniciando o Teste em Série...\n')

# ============================================================
# LOOP PRINCIPAL DOS TESTES
# ============================================================
start_time = time.time()
for nr_test, img in enumerate(remaining_imgs):
    training_time = time.time()
    #print(f"--- Teste Nr {nr_test + 1}/{len(remaining_imgs)} ---")
    
    # ============================================================
    # DADOS DE ENTRADA
    # ============================================================
    #Extrai uma amostra de dados balanceada
    ch_url_dict['Mask'] = os.path.join(Temp_Imgs_Folder, 'Mask')
    Img_train, Label_train, Scaler_ds, _ = prepare_dataset_vs2(ch_url_dict, img, NS, channel_list, window_size=WindowSize, balance_ratio=0.5)
    # print(f'Selecionadas {len(Img_train)} amostras de treino.')
    
    #Extrai todas as amostras positivas da imagem original
    ch_url_dict['Mask'] = os.path.join(DS_DIR, ref_mask)
    Img_full, Label_full, Scale, coords_full = prepare_dataset_vs2(ch_url_dict, img, NS, channel_list, window_size=WindowSize, balance_ratio=0, scaler_ds = Scaler_ds)
    
    # ============================================================
    # TREINAMENTO INICIAL DO SOM
    # ============================================================
    som = MiniSom(
        x=MS[0], y=MS[1],
        input_len=len(Img_train[0]),
        sigma=sigma_f,
        learning_rate=l_r,
        neighborhood_function=ngbh_function,
        topology=topology
    )
    som.random_weights_init(Img_train)
    som.train_random(Img_train, num_iteration=1000)
    # print("Treinamento inicial (fase de ordenação topológica) concluído\n")

    # ============================================================
    # AJUSTE FINO (FINE-TUNING)
    # ============================================================
    q_error, t_error = [], []
    # print("Iniciando ajuste fino com monitoramento de QE e TE...")
    for i in range(epochs):
        som.train_random(Img_train, batch_size)
        q_err = som.quantization_error(Img_train)
        t_err = som.topographic_error(Img_train)
        q_error.append(q_err)
        t_error.append(t_err)
        # if i % 10 == 0 or i == epochs - 1:
        #     print(f"Iteração {i+1}/{epochs} | QE={q_err:.4f} | TE={t_err:.4f}")

    elapsed_time = time.time() - training_time
    # print(f"Tempo de treinamento: {elapsed_time/60:.2f} min\n")
    # ================================================
    # Probabilidade observada (likelihood base)
    # ================================================
    # print(f'Calculando a probabilidade a priori da imagem {img}. \n')
    prob_map, neuron_classes = PrioriProb (Img_train, Label_train, som, MS[0], MS[1])
    
    # ================================================
    # Estimativa Bayesiana
    # ================================================
    # print(f'Calculando a probabilidade a posteriori da imagem {img}. \n')
    posterior_map = PostProb (MS[0], MS[1], prob_map)
    
    # ================================================
    # Visualização Mapas de Probabilidade
    # ================================================

    # fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # # Mapa a priori
    # im0 = axes[0].imshow(prob_map, cmap='viridis', vmin=0, vmax=1)
    # axes[0].set_title("Prior probability map")  #Mapa de Probabilidade A Priori
    # #axes[0].set_xlabel("Eixo Y do SOM")
    # #axes[0].set_ylabel("Eixo X do SOM")
    # fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)#, label="P(estrada)")
    
    # # Mapa a posteriori
    # im1 = axes[1].imshow(posterior_map, cmap='viridis', vmin=0, vmax=1)
    # axes[1].set_title("Conditional probability map") #Mapa de Probabilidade A Posteriori
    # #axes[1].set_xlabel("Eixo Y do SOM")
    # #axes[1].set_ylabel("Eixo X do SOM")
    # fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)#, label="P(estrada)")
    # plt.savefig(os.path.join(OutputFolder, f"Prob_Maps_{img}.png"), dpi=300, bbox_inches="tight")
    # plt.tight_layout()
    # plt.show()
    
    # ============================================================
    # FILTRA AS AMOSTRAS DO DATASET ORIGINAL POR UMA REDE SOM
    # ============================================================
    #print(f'Filtrando mascara original {img}... \n')
    #X_keep, y_keep, X_remove, y_remove, X_flag, y_flag, keep_idx, remove_idx, flag_idx = CheckSamples(som, prob_map, posterior_map, Img_full, Label_full, t_c = 0.7, t_p = 0.7)
    X_keep_pos, y_keep_pos, X_remove_pos, y_remove_pos, keep_idx, remove_idx, updated_labels = CheckSamples_vs1(som, prob_map, posterior_map, Img_full, Label_full, t_c = 0.6, t_p = 0.6)
    #X_keep_pos, y_keep_pos, X_remove_pos, y_remove_pos, X_added, y_added, kept_pos_idx, removed_pos_idx, added_idx, updated_labels = CheckSamples_vs2(som, prob_map, posterior_map, Img_full, Label_full, t_c = 0.6, t_p = 0.6)
                                                                                          
    #print(f"Amostras mantidas: {len(X_keep_pos)}")
    #print(f"Amostras removidas: {len(X_remove_pos)}\n")
    #print(f"Amostras sinalizadas: {len(X_flag)}")

    # =======================================
    # IMPRIME A MASCARA ORIGINAL E A FILTRADA
    # =======================================

    with rasterio.open(os.path.join(ch_url_dict['Mask'], img)) as src:
        mask_original = src.read(1)
        meta_clean = src.meta.copy()
    
        # limpar campos problemáticos
        for key in ["tiled", "compress", "interleave",
                    "photometric", "blockxsize", "blockysize"]:
            if key in meta_clean:
                meta_clean.pop(key)
        
        meta_clean.update({
            "dtype": "uint8",
            "count": 1
        })
        
        h, w = mask_original.shape
        mask_filtered = updated_labels.reshape(h, w)
        

        output_file = os.path.join(masks_filtered_Folder, img)
        with rasterio.open(output_file, "w", **meta_clean) as dst:
            dst.write(mask_filtered.astype(np.uint8), 1)
        
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(mask_original, cmap='gray')
        # plt.title('Máscara Original')
        # plt.axis('off')
        
        # plt.subplot(1, 2, 2)
        # plt.imshow(mask_filtered, cmap='gray')
        # plt.title('Máscara Filtrada (SOM)')
        # plt.axis('off')
        
        # plt.suptitle(f"Comparação - {img}", fontsize=14)
        # plt.tight_layout()
        # plt.savefig(os.path.join(OutputFolder, f"Mask_Comparison_{img}.png"), dpi=200, bbox_inches="tight")
        # plt.show()
    
    # ============================================================
    # AVALIAÇÃO DE DESEMPENHO
    # ============================================================
        
    # Atualiza melhor modelo
    if best_qe == 0:
        best_qe = q_err
        Best_Model_QE = q_err
        Best_Model_TE = t_err
        best_model = som
    elif best_qe > q_err:
        best_qe = q_err
        Best_Model_QE = q_err
        Best_Model_TE = t_err
        best_model = som
        # ============================================================
        # AVALIAÇÃO DE CONVERGÊNCIA DO MELHOR MODELO
        # ============================================================
        plt.figure(figsize=(8,4))
        plt.plot(q_error, label='Erro de Quantização (QE)')
        plt.plot(t_error, label='Erro Topográfico (TE)')
        plt.xlabel('Época')
        plt.ylabel('Erro')
        plt.title(f'Convergência SOM - Ajuste fino - - Imagem {img}')
        plt.legend()
        plt.savefig(os.path.join(OutputFolder, f'Convergência SOM - Ajuste fino - Teste{nr_test + 1} - Imagem {img}'), dpi=150, bbox_inches="tight")
        plt.grid(True)
        plt.show()
        print("Best_QE: ", q_err)


    elapsed_time = time.time() - training_time
    print(f"- Teste {nr_test + 1}/{len(remaining_imgs)} - Imagem {img} - QE {q_err:.4f} - TE {t_err:.4f} - Amostras mantidas: {len(X_keep_pos)} - Amostras removidas: {len(X_remove_pos)}- Tempo de treinamento: {elapsed_time/60:.2f} min\n")

    #acrescenta a imagem à lista de processadas
    processed_imgs.append(img)

elapsed_time = time.time() - start_time
print(f"Tempo de treinamento: {elapsed_time/60:.2f} min\n")

print('#==================================================================\n')
print('Procedimento concluído com sucesso')
print('best_qe: ', best_qe)
print(f'{len(processed_imgs)} imagens processadas: \n')