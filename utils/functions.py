#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:15:38 2026

@author: usuario
"""

############################################## FUNCTIONS ####################################################

#Função que extrai os esqueletos das estradas
def skeletonize_mask(mask, im_show=True):
    # Normaliza para booleano
    mask_bool = mask > 0
    skel = skeletonize(mask_bool)

    if im_show:
        # Plotagem
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
   
        axes[0].imshow(mask, cmap='gray')
        axes[0].set_title("Mascara Original"); axes[0].axis("off")
    
        axes[1].imshow(skel, cmap='gray')
        axes[1].set_title("Mascara Esqueletizada"); axes[1].axis("off")
    
        #fig.savefig(os.path.join(output_folder, f"Teste_{img}.png"))
        plt.show()
        plt.close(fig)

    # Converte para uint8
    return (skel.astype(np.uint8))

#Recorta as imagens segundo um formato pré-definido
def crop_image(img, new_size):
    lin,col = img.size
    
    # Calcular as coordenadas para o corte central
    left = (col - new_size) // 2
    top = (col - new_size) // 2   
    right = left + new_size       
    bottom = top + new_size       
    
    # Recortar a imagem
    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img

def crop_and_save_image(file, input_path, output_path, NS, is_mask=False):
    """
    Abre uma imagem, recorta e salva no formato adequado.
    
    Args:
        file (str): Nome do arquivo.
        input_path (str): Caminho da imagem de entrada.
        output_path (str): Caminho de saída.
        NS (tuple): Parâmetros de recorte.
        Filter (bool): (opcional) aplicar filtro.
        is_mask (bool): Se True, força salvamento como uint8 (para OpenCV).
    """
    in_file = os.path.join(input_path, file)
    
    # Ler imagem — usa Pillow (funciona bem para máscaras 1 banda)
    img = Image.open(in_file)
    cropped_img = crop_image(img, NS[0])  # sua função de recorte
    
    img_array = np.array(cropped_img)

    # === Ajuste automático do tipo ===
    if is_mask:
        # garante compatibilidade com cv2.imread()
        if img_array.max() <= 1:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    #else:
        ## imagens multibanda — mantém precisão
        #if img_array.dtype != np.float32:
        #    img_array = img_array.astype(np.float32)

    # Salvar com tipo adequado
    tiff.imwrite(os.path.join(output_path, file), img_array)
    
def extract_features_window_vs0(image, window_size=3):
    # --- Ajusta formato para (H, W, C) 
    if image.ndim == 2: 
        image = image[..., np.newaxis] 
    elif image.shape[0] < image.shape[-1]: 
        # (C, H, W) 
        image = np.transpose(image, (1, 2, 0)) 
        
    H, W, C = image.shape 
    pad = window_size // 2 
    
    # --- Padding refletido para bordas 
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect') 
    
    # --- Extrai janelas (H, W, window_size, window_size, C) 
    patches = view_as_windows(padded, (window_size, window_size, C)) 
    
    # --- Reorganiza: (H, W, window_size*window_size*C) 
    patches = patches.reshape(H, W, -1) 
    
    # --- Achata para (N_pixels, N_features) 
    features = patches.reshape(-1, patches.shape[-1]) 
    
    return features

def extract_features_window(image, window_size_y=3, window_size_x=None):
    """
    Extrai vetores de características baseados em janelas (ny × nx)
    centradas em cada pixel. Pode ser quadrada (ny=nx) ou linear (ny=1 ou nx=1).

    Parâmetros
    ----------
    image : np.ndarray
        Imagem de entrada. Pode ter formato:
            - (H, W)
            - (C, H, W) ou (H, W, C)
    window_size_y : int
        Altura da janela (número de linhas)
    window_size_x : int, opcional
        Largura da janela (número de colunas). Se None, assume window_size_y.

    Retorna
    -------
    features : np.ndarray
        Matriz 2D com dimensões (N_pixels_validos, N_features)
    """

    if window_size_x is None:
        window_size_x = window_size_y

    # --- Ajusta formato para (H, W, C)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.shape[0] < image.shape[-1]:  # (C, H, W)
        image = np.transpose(image, (1, 2, 0))

    H, W, C = image.shape
    pad_y, pad_x = window_size_y // 2, window_size_x // 2

    # --- Padding refletido
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode='reflect')

    # --- Extrai janelas (H, W, wy, wx, C)
    patches = view_as_windows(padded, (window_size_y, window_size_x, C))

    # --- Reorganiza
    patches = patches.reshape(H, W, -1)
    features = patches.reshape(-1, patches.shape[-1])

    return features

def extract_features(image):
    # Verificar o número de canais da imagem
    if len(image.shape) == 2:  # Imagem de 1 canal
        features = image.flatten()  # Achatar para um vetor 1D
    elif len(image.shape) == 3:  # Imagem de mais de 1 canal (RGB)
        # Combinar os canais em um único vetor de características
        # --- Reorganiza para (512, 512, 6)
        image = np.transpose(image, (1, 2, 0))
        features = image.reshape (-1, image.shape[2])

    return features

def prepare_dataset_vs0(imgs_url_dict, Imgs_List, Labels_List, n_total, NS, ch_list, t_size=0.1):

    Imgs_train, Imgs_test, y_train, y_test = train_test_split(Imgs_List, Labels_List, test_size=t_size)
    
    print(" ")
    print(f"Conjunto de treinamento: {len(Imgs_train)} imagens")
    #print(f"Conjunto de validação: {len(X_val)} imagens")
    print(f"Conjunto de teste: {len(Imgs_test)} imagens")
    
    # 2. PROCESSAR AS IMAGENS DE TREINAMENTO
    Img_train = []
    Label_train = []
    total_features = 0
    
    #calcula a qtd de imagens para compor o total de amostras desejado
    nr_imgs = int(n_total/(NS[0]*NS[1])) + 1
    
    # Seleciona um cjt aleatorio de imagens para compor o dataset
    indices = random.sample(range(len(Imgs_List)), nr_imgs)
            
    # Seleciona os mesmos elementos em ambas as listas
    Imgs_Selected = [Imgs_List[i] for i in indices]
    Labels_Selected = [Labels_List[i] for i in indices]
    print(f'- Selecionadas {len(Imgs_Selected)} imagens.')
    
    #for img in tqdm(Imgs_train, desc="Preparando o dataset")
    for idx, img in enumerate(Imgs_Selected):
        #update the mask paths
        label_path = os.path.join(imgs_url_dict['Mask'], img)
        
        #mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(label_path))
        #transforma as mascaras em matriz binaria
        mask_bin = np.where(mask > 0, 1, 0)
        
        #verificacao da mascara
        #print('Valores únicos:', np.unique(mask_bin))
        #print('Proporção de estrada:', np.sum(mask_bin == 1) / mask_bin.size)
        
        #mask_bin = skeletonize(mask_bin)
        mask_flatten = mask_bin.flatten()

        #plt.imshow(mask_bin*255, cmap = 'gray')
        #plt.axis("off")  # Oculta os eixos
        #plt.show()
    
        # Cria as imagens empilhadas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.uint16)
        
        for i, ch in enumerate(ch_list):
            img_path = os.path.join(imgs_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.uint16)
            
            # Verifica tamanho
            assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img}: {b_img.shape}"
            
            image[i, :, :] = b_img
        
        features = extract_features(image)
        
        Img_train.append(features)
        Label_train.append(mask_flatten)

        
    # Converter para arrays
    if IMG_CHANNELS > 1:
        Img_train = np.vstack(Img_train) #empilhamento vertical, aumenta o numero de linhas (* em uma np array cada linha é um cjt de dados entre um [])
        Label_train = np.hstack(Label_train) #empilhamento horizontal, aumenta o numero de colunas (* em uma np array cada coluna é um numero dentro de um [])
    else:
        Img_train = np.array(Img_train)
        Label_train = np.array(Label_train)
        Img_train = Img_train.reshape(-1,1)
        Label_train = Label_train.reshape(-1,1)
        Label_train = Label_train.astype(np.uint8)
        
    return Img_train, Label_train, Imgs_test, y_test


def prepare_dataset_vs1(imgs_url_dict, Imgs_List, Labels_List, NS, ch_list, t_size=0.2):

    # 1. PROCESSAR AS IMAGENS DE TREINAMENTO
    Img_ds = []
    Label_ds = []
    
    print(f'- Selecionadas {len(Imgs_List)} imagens.')
    
    #preparo do dataset
    for idx, img in enumerate(Imgs_List):
        #update the mask paths
        label_path = os.path.join(imgs_url_dict['Mask'], img)
        
        #mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(Image.open(label_path))
        #transforma as mascaras em matriz binaria
        mask_bin = np.where(mask > 0, 1, 0)
        #mask_bin = skeletonize(mask_bin)
        mask_flatten = mask_bin.flatten()
    
        # Cria as imagens empilhadas
        image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.uint16)
        
        for i, ch in enumerate(ch_list):
            img_path = os.path.join(imgs_url_dict[ch], img)
            b_img = tiff.imread(img_path).astype(np.uint16)
            
            # Verifica tamanho
            assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img}: {b_img.shape}"
            
            image[i, :, :] = b_img
        
        features = extract_features(image)
        
        Img_ds.append(features)
        Label_ds.append(mask_flatten)

        
    # Converter para arrays
    if IMG_CHANNELS > 1:
        Img_ds = np.vstack(Img_ds) #empilhamento vertical, aumenta o numero de linhas (* em uma np array cada linha é um cjt de dados entre um [])
        Label_ds = np.hstack(Label_ds) #empilhamento horizontal, aumenta o numero de colunas (* em uma np array cada coluna é um numero dentro de um [])
    else:
        Img_ds = np.array(Img_ds)
        Img_ds = Img_ds.reshape(-1,1)
        
        Label_ds = np.array(Label_ds)
        Label_ds = Label_ds.reshape(-1,1)
        Label_ds = Label_ds.astype(np.uint8)
        
    # Escalar as características
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_ds = scaler.fit_transform(Img_ds)
    
    X_train, X_test, y_train, y_test = train_test_split(X_ds, Label_ds, test_size=t_size)
    
    print(" ")
    print(f"Conjunto de treinamento: {len(X_train)} pontos")
    print(f"Conjunto de teste: {len(X_test)} pontos")
        
    return X_train, y_train, X_test, y_test, scaler


def prepare_dataset_vs2(imgs_url_dict, img_name, NS, ch_list, window_size=(3,3), balance_ratio=0.5, scaler_ds=False):
    """
    Prepara o dataset de uma imagem e sua máscara correspondente,
    extraindo vetores de características locais e, opcionalmente, balanceando as amostras.

    Parâmetros
    ----------
    imgs_url_dict : dict
        Dicionário com caminhos das bandas e da máscara.
        Ex: {'B8': '...', 'B11': '...', 'NVI': '...', 'Mask': '...'}
    img_name : str
        Nome do arquivo da imagem base.
    NS : tuple
        Dimensões (altura, largura) da imagem.
    ch_list : list
        Lista de chaves das bandas no dicionário imgs_url_dict.
    window_size : int, opcional
        Tamanho da janela de vizinhança (default=5).
    balance_ratio : float, opcional
        Proporção desejada de amostras positivas/negativas (default=0.5 → 1:1).
        Se balance_ratio=0, retorna todas as amostras.
    scaler_ds : sklearn.preprocessing.MinMaxScaler, opcional
        Escalonador previamente ajustado (para manter consistência entre treino e teste).

    Retorna
    -------
    X_scaled : np.ndarray
        Vetores de características escalonados.
    y : np.ndarray
        Máscara correspondente (0=fundo, 1=estrada).
    scaler : MinMaxScaler
        Escalonador usado.
    coords : np.ndarray
        Coordenadas (linha, coluna) dos pixels correspondentes aos vetores.
    """

    # ======================
    # 1️Ler máscara binária
    # ======================
    label_path = os.path.join(imgs_url_dict['Mask'], img_name)
    mask = np.array(Image.open(label_path))
    mask_bin = (mask > 0).astype(np.uint8)
    mask_flatten = mask_bin.flatten()

    # ======================
    # 2️ Empilhar bandas
    # ======================
    image = np.zeros((len(ch_list), NS[0], NS[1]), dtype=np.uint16)
    for i, ch in enumerate(ch_list):
        img_path = os.path.join(imgs_url_dict[ch], img_name)
        b_img = tiff.imread(img_path).astype(np.uint16)
        assert b_img.shape == (NS[0], NS[1]), f"Tamanho incompatível em {ch}/{img_name}: {b_img.shape}"
        image[i, :, :] = b_img

    # ======================
    # 3️ Extrair características locais
    # ======================
    # print('window_size[0]: ', window_size[0])
    # print('window_size[1] ', window_size[1])
    features = extract_features_window(image, window_size[0], window_size[1])
    #features = extract_features_window(image, window_size = window_size)
    #features = extract_features_window_diff(image, window_size=window_size)

    # Cria coordenadas para cada pixel válido
    rows, cols = np.indices((NS[0], NS[1]))
    coords = np.column_stack((rows.flatten(), cols.flatten()))

    # ======================
    # 4️ Balancear amostras
    # ======================
    if balance_ratio != 0:
        idx_pos = np.where(mask_flatten == 1)[0]
        idx_neg = np.where(mask_flatten == 0)[0]
    
        n_pos = len(idx_pos)
        n_neg = len(idx_neg)
        if n_pos == 0:
            raise ValueError(f"Nenhum pixel positivo encontrado em {img_name}.")

        # Número de negativos selecionados proporcionalmente
        n_neg_sel = int(n_pos / balance_ratio - n_pos) if balance_ratio < 0.5 else n_pos
        n_neg_sel = min(n_neg_sel, n_neg)

        np.random.seed(42)
        idx_neg_sel = np.random.choice(idx_neg, size=n_neg_sel, replace=False)

        idx_final = np.concatenate([idx_pos, idx_neg_sel])
        np.random.shuffle(idx_final)

        X = features[idx_final]
        y = mask_flatten[idx_final]
        coords = coords[idx_final]
    else:
        X = features
        y = mask_flatten

    # ======================
    # 5️ Escalonar vetores
    # ======================
    if not scaler_ds:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        scaler = scaler_ds
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, coords

# ============================================================
# PERCORRER TODAS AS AMOSTRAS DO DATASET
# ============================================================
#verifica as amostras de uma mascara e elimina as que não passaram pelos criterios
def CheckSamples(som_map, prob_map_1, posterior_map, data_scaled, labels, t_c = 0.6, t_p = 0.6):
    # Listas para armazenar índices das amostras
    keep_idx = []
    remove_idx = []
    flag_idx = []

    for i, sample in enumerate(data_scaled):
        neuron = som_map.winner(sample)
    
        # Obtém probabilidades associadas ao neurônio vencedor
        prior_prob = prob_map_1[neuron]          # probabilidade observada (classe 1)
        posterior_prob = posterior_map[neuron]   # probabilidade ajustada (classe 1)
    
        # ========================================================
        # APLICA AS REGRAS DE DECISÃO (Santos, 2021)
        # ========================================================
    
        if prior_prob < t_c:
            # Caso (a): provável ruído de classe
            remove_idx.append(i)
    
        elif prior_prob >= t_c and posterior_prob >= t_p:
            # Caso (b): amostra confiável
            keep_idx.append(i)
    
        else:
            # Caso intermediário (variação real de uso/cobertura)
            flag_idx.append(i)
    
    # ============================================================
    # 4️CRIA OS SUBCONJUNTOS FILTRADOS
    # ============================================================
    
    X_keep = data_scaled[keep_idx]
    y_keep = np.array(labels)[keep_idx]
    
    X_remove = data_scaled[remove_idx]
    y_remove = np.array(labels)[remove_idx]
    
    X_flag = data_scaled[flag_idx]
    y_flag = np.array(labels)[flag_idx]

    return X_keep, y_keep, X_remove, y_remove, X_flag, y_flag, keep_idx, remove_idx, flag_idx


def CheckSamples_vs1(som_map, prob_map_1, posterior_map, data_scaled, labels, t_c=0.6, t_p=0.6):
    """
    Filtra apenas as amostras positivas (classe 1) com base nas
    probabilidades a priori e a posteriori, removendo pixels não confiáveis.

    Parâmetros
    ----------
    som_map : MiniSom
        Rede SOM já treinada.
    prob_map_1 : np.ndarray
        Mapa de probabilidades observadas (priori) para classe 1.
    posterior_map : np.ndarray
        Mapa de probabilidades ajustadas (posteriori) para classe 1.
    data_scaled : np.ndarray
        Vetores de características escalonados (todas as amostras).
    labels : np.ndarray
        Vetor de rótulos (0=fundo, 1=estrada).
    t_c : float
        Limite inferior para probabilidade observada (priori).
    t_p : float
        Limite inferior para probabilidade posteriori.

    Retorna
    -------
    X_keep : np.ndarray
        Amostras positivas mantidas.
    y_keep : np.ndarray
        Labels correspondentes (1).
    X_remove : np.ndarray
        Amostras positivas removidas.
    y_remove : np.ndarray
        Labels correspondentes (1).
    keep_idx : list
        Índices das amostras positivas mantidas.
    remove_idx : list
        Índices das amostras positivas removidas.
    """

    # Índices das amostras positivas
    pos_idx = np.where(np.array(labels) == 1)[0]

    keep_idx = []
    remove_idx = []

    for i in pos_idx:
        sample = data_scaled[i]
        neuron = som_map.winner(sample)

        # Probabilidades associadas ao neurônio vencedor
        prior_prob = prob_map_1[neuron]
        posterior_prob = posterior_map[neuron]

        # ========================================================
        # Regras de decisão (aplicadas apenas às amostras positivas)
        # ========================================================
        if prior_prob < t_c or posterior_prob < t_p:
            # Caso de baixa confiança → remover
            remove_idx.append(i)
        else:
            # Caso confiável → manter
            keep_idx.append(i)

    # ============================================================
    # Cria subconjuntos filtrados (somente positivos)
    # ============================================================
    X_keep = data_scaled[keep_idx]
    y_keep = np.array(labels)[keep_idx]

    X_remove = data_scaled[remove_idx]
    y_remove = np.array(labels)[remove_idx]
    
    
    # ----------------------------------------------------------
    # Construir máscara final
    # ----------------------------------------------------------
    updated_labels = labels.copy()
    updated_labels[remove_idx] = 0  # remover positivos

    return X_keep, y_keep, X_remove, y_remove, keep_idx, remove_idx, updated_labels

def CheckSamples_vs2(som_map, prob_map_1, posterior_map, data_scaled, labels, t_c=0.6, t_p=0.6):
    """
    Analisa as amostras negativas: se ambas as probabilidades
    (priori e posteriori) forem >= limiares, o pixel é promovido à classe 1.

    Parâmetros
    ----------
    som_map : MiniSom
        Rede SOM treinada.
    prob_map_1 : np.ndarray
        Probabilidades observadas (priori) para classe 1.
    posterior_map : np.ndarray
        Probabilidades ajustadas (posteriori) para classe 1.
    data_scaled : np.ndarray
        Todas as amostras (normalizadas/escaladas).
    labels : np.ndarray
        Máscara original (0=fundo, 1=estrada).
    t_c : float
        Limiar para probabilidade priori.
    t_p : float
        Limiar para probabilidade posteriori.

    Retorna
    -------
    X_keep_pos : np.ndarray
        Amostras positivas que permaneceram positivas.
    y_keep_pos : np.ndarray

    X_remove_pos : np.ndarray
        Amostras positivas removidas por baixa confiança.
    y_remove_pos : np.ndarray

    X_added : np.ndarray
        Amostras negativas que foram promovidas a positivas.
    y_added : np.ndarray (=1)

    kept_pos_idx : list
    removed_pos_idx : list
    added_idx : list

    updated_labels : np.ndarray
        Máscara final após remoções e adições
    """

    labels = np.array(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    kept_pos_idx = []
    removed_pos_idx = []
    added_idx = []

    # ----------------------------------------------------------
    # 1) Avaliar as amostras positivas (como antes)
    # ----------------------------------------------------------
    for i in pos_idx:
        sample = data_scaled[i]
        neuron = som_map.winner(sample)

        prior_prob = prob_map_1[neuron]
        post_prob = posterior_map[neuron]

        if prior_prob < t_c or post_prob < t_p:
            removed_pos_idx.append(i)
        else:
            kept_pos_idx.append(i)

    # ----------------------------------------------------------
    # 2) Avaliar amostras negativas (para possível promoção)
    # ----------------------------------------------------------
    for i in neg_idx:
        sample = data_scaled[i]
        neuron = som_map.winner(sample)

        prior_prob = prob_map_1[neuron]
        post_prob = posterior_map[neuron]

        # Condição de promoção para classe 1
        if prior_prob >= 0.9 and post_prob >= 0.9:
            added_idx.append(i)

    # ----------------------------------------------------------
    # Construir subconjuntos
    # ----------------------------------------------------------
    X_keep_pos = data_scaled[kept_pos_idx]
    y_keep_pos = labels[kept_pos_idx]

    X_remove_pos = data_scaled[removed_pos_idx]
    y_remove_pos = labels[removed_pos_idx]

    X_added = data_scaled[added_idx]
    y_added = np.ones(len(added_idx), dtype=int)

    # ----------------------------------------------------------
    # Construir máscara final
    # ----------------------------------------------------------
    updated_labels = labels.copy()
    updated_labels[removed_pos_idx] = 0  # remover positivos
    updated_labels[added_idx] = 1        # adicionar novos positivos

    return (
        X_keep_pos, y_keep_pos,
        X_remove_pos, y_remove_pos,
        X_added, y_added,
        kept_pos_idx, removed_pos_idx, added_idx,
        updated_labels
    )


# ================================================
# 1️⃣ Probabilidade observada (likelihood base)
# ================================================
def PrioriProb (data_scaled, labels, som, map_x, map_y):
    neuron_classes = {}
    for i, sample in enumerate(data_scaled):
        winner = som.winner(sample)
        if winner not in neuron_classes:
            neuron_classes[winner] = []
        neuron_classes[winner].append(labels[i])
    
    prob_map = np.zeros((map_x, map_y))
    
    for neuron, classes in neuron_classes.items():
        cont_class = Counter(classes)
        n_ones = cont_class.get(1.0, 0)
        n_zeros = cont_class.get(0.0, 0)
        t_samples = n_ones + n_zeros
    
        if t_samples > 0:
            p_road = n_ones / t_samples  # probabilidade observada
            prob_map[neuron] = p_road

    return prob_map, neuron_classes

# ================================================
# 2️⃣ Estimativa Bayesiana
# ================================================

def PostProb (map_x, map_y, prob_map):
    '''Calcula a probabilidade posterior (Bayesiana) de um neurônio em uma rede SOM pré-treinada'''

    # Cria uma cópia para armazenar o mapa posterior
    posterior_map = np.zeros_like(prob_map)
    
    # Função para pegar vizinhos (4 ou 8 conexões)
    def get_neighbors(x, y, map_x, map_y, radius=1):
        neighbors = []
        for i in range(x - radius, x + radius + 1):
            for j in range(y - radius, y + radius + 1):
                if (i, j) != (x, y) and 0 <= i < map_x and 0 <= j < map_y:
                    neighbors.append((i, j))
        return neighbors
    
    # Percorre cada neurônio e aplica a inferência bayesiana
    for x in range(map_x):
        for y in range(map_y):
            y_jk = prob_map[x, y]  # probabilidade observada
            neighbors = get_neighbors(x, y, map_x, map_y)
            neigh_values = [prob_map[i, j] for i, j in neighbors]# if prob_map[i, j] > 0]
    
            # Se não houver vizinhos válidos, mantém valor original
            if len(neigh_values) == 0:
                posterior_map[x, y] = y_jk
                continue
    
            # PRIOR: média e variância dos vizinhos
            m_jk = np.mean(neigh_values)
            s2_jk = np.var(neigh_values)
    
            # Hiperparâmetro σ²
            sigma2_j = abs(0.999999 - max(y_jk, 1e-6))  # evita zero
    
            # POSTERIOR: fórmula (3.9)
            E_theta = (m_jk * sigma2_j + y_jk * s2_jk) / (sigma2_j + s2_jk)
            posterior_map[x, y] = E_theta

    return posterior_map

#Road Index
def S2_RI(banda_b2_path, banda_nir_path, banda_swir_path, ri_saida_path):
    """ Implementa o índice de realce de estradas em bandas Sentinel-2, conforme apresentado pelos autores 
        Muhammad Waqas Ahmed, * , Sumayyah Saadi, Muhammad Ahmed no artigo 
        'Automated road extraction using reinforced road indices for Sentinel-2 data'"""
    # Abrir as bandas
    with rasterio.open(banda_b2_path) as b_blue, rasterio.open(banda_nir_path) as b_nir, rasterio.open(banda_swir_path) as b_swir:
        # Ler os dados das bandas como arrays NumPy
        b2 = b_blue.read(1).astype(np.float32)  # Banda 2 (Blue)
        nir = b_nir.read(1).astype(np.float32)  # Banda 8 (NIR)
        swir = b_swir.read(1).astype(np.float32)  # Banda 11 (SWIR)
    
        # Evitar divisão por zero usando np.errstate
        #NDBI = (B8 - B11) / (B8 + B11)
        
        # Evitar divisões por zero
        den = (swir + nir + b2).astype(np.float32)
        den[den == 0] = np.nan  # evita problemas
        
        # Calcula o índice
        RI = 1 - (3.0 * np.minimum.reduce([swir, nir, b2])) / den
        
        # Substituir valores inválidos
        RI[np.isnan(RI)] = 0  # Substituir NaN por 0
        RI[np.isinf(RI)] = 0  # Substituir infinitos por 0
        
        # # Opcional: normalizar valores (caso queira em [0,1])
        # RI = np.clip(RI, 0, 1)
    
        # Criar perfil para o arquivo de saída (mantém referência espacial)
        profile = b_nir.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-1) #nodata=0
    
        # Salvar o NDBI como TIFF
        with rasterio.open(ri_saida_path, "w", **profile) as dst:
            dst.write(RI, 1)

#NDRI index
def S2_NDRI(banda_b2_path, banda_nir_path, ndri_saida_path):
    """ Implementa uma proposta do índice de realce de estradas em bandas Sentinel-2'"""
    # Abrir as bandas
    with rasterio.open(banda_b2_path) as b_blue, rasterio.open(banda_nir_path) as b_nir:
        # Ler os dados das bandas como arrays NumPy
        b2 = b_blue.read(1).astype(np.float32)  # Banda 2 (Blue)
        nir = b_nir.read(1).astype(np.float32)  # Banda 8 (NIR)
    
        # Evitar divisão por zero usando np.errstate
        #NDRI = (NIR- B2) / (NIR + B2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ndri = (nir - b2) / (b2 + nir)
            
        # Substituir valores inválidos
        ndri[np.isnan(ndri)] = 0  # Substituir NaN por 0
        ndri[np.isinf(ndri)] = 0  # Substituir infinitos por 0
        
        ## Garantir que os valores fiquem entre -1 e 1
        #ndri = np.clip(ndbi, -1, 1)
        
        # Criar perfil para o arquivo de saída (mantém referência espacial)
        profile = b_nir.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-1) #nodata=0
    
        # Salvar o NDBI como TIFF
        with rasterio.open(ndri_saida_path, "w", **profile) as dst:
            dst.write(ndri, 1)

#NDBI index
def S2_NDBI(banda_nir_path, banda_swir_path, ndbi_saida_path):
    # Abrir as bandas
    with rasterio.open(banda_nir_path) as b_nir, rasterio.open(banda_swir_path) as b_swir:
        # Ler os dados das bandas como arrays NumPy
        nir = b_nir.read(1).astype(np.float32)  # Banda 8 (NIR)
        swir = b_swir.read(1).astype(np.float32)  # Banda 11 (SWIR)
    
        # Evitar divisão por zero usando np.errstate
        #NDBI = (B8 - B11) / (B8 + B11)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndbi = (nir - swir) / (swir + nir)
            
        # Substituir valores inválidos
        ndbi[np.isnan(ndbi)] = 0  # Substituir NaN por 0
        ndbi[np.isinf(ndbi)] = 0  # Substituir infinitos por 0
        
        ## Garantir que os valores fiquem entre -1 e 1
        #ndbi = np.clip(ndbi, -1, 1)
    
        # Criar perfil para o arquivo de saída (mantém referência espacial)
        profile = b_nir.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-1) #nodata=0
    
        # Salvar o NDBI como TIFF
        with rasterio.open(ndbi_saida_path, "w", **profile) as dst:
            dst.write(ndbi, 1)
#NDVI index
def S2_NDVI(banda_nir_path, banda_red_path, ndvi_saida_path):
    # Abrir as bandas
    with rasterio.open(banda_nir_path) as b_nir, rasterio.open(banda_red_path) as b_red:
        # Ler os dados das bandas como arrays NumPy
        nir = b_nir.read(1).astype(np.float32)  # Banda 8 (NIR)
        red = b_red.read(1).astype(np.float32)  # Banda 4 (Red)
    
        #Evitar divisão por zero usando np.errstate
        # NDVI = (NIR - RED) / (NIR + RED) = (B8 - B4) / (B8 + B4)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (red + nir)
            
        # Substituir valores inválidos
        ndvi[np.isnan(ndvi)] = 0  # Substituir NaN por 0
        ndvi[np.isinf(ndvi)] = 0  # Substituir infinitos por 0
        
        ## Garantir que os valores fiquem entre -1 e 1
        #ndbi = np.clip(ndbi, -1, 1)
    
        # Criar perfil para o arquivo de saída (mantém referência espacial)
        profile = b_nir.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-1) #nodata=0
    
        # Salvar o NDVI como TIFF
        with rasterio.open(ndvi_saida_path, "w", **profile) as dst:
            dst.write(ndvi, 1)

#NDWI index
def S2_NDWI(banda_nir_path, banda_green_path, ndwi_saida_path):
    # Abrir as bandas
    with rasterio.open(banda_nir_path) as b_nir, rasterio.open(banda_green_path) as b_green:
        # Ler os dados das bandas como arrays NumPy
        nir = b_nir.read(1).astype(np.float32)  # Banda 8 (NIR)
        green = b_green.read(1).astype(np.float32)  # Banda 3 (Green)
    
        #Evitar divisão por zero usando np.errstate
        # NDWI = (G - NIR) / (G + NIR) = (B03 - B08) / (B03 + B08)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (nir + green)
            
        # Substituir valores inválidos
        ndwi[np.isnan(ndwi)] = 0  # Substituir NaN por 0
        ndwi[np.isinf(ndwi)] = 0  # Substituir infinitos por 0
        
        ## Garantir que os valores fiquem entre -1 e 1
        #ndbi = np.clip(ndbi, -1, 1)
    
        # Criar perfil para o arquivo de saída (mantém referência espacial)
        profile = b_nir.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=-1) #nodata=0
    
        # Salvar o NDVI como TIFF
        with rasterio.open(ndwi_saida_path, "w", **profile) as dst:
            dst.write(ndwi, 1)
            

def SOM_Metrics(SOM, data_scaled, labels, plot_cm = False):
    # MATRIZ DE CONFUSÃO

    # ============================================================
    # MAPEAMENTO DOS NEURÔNIOS PARA CLASSES
    # ============================================================
    neuron_classes = {}
    for i, sample in enumerate(data_scaled):
        winner = SOM.winner(sample)
        if winner not in neuron_classes:
            neuron_classes[winner] = []
        neuron_classes[winner].append(labels[i])
    
    # Mapa de classes majoritárias de cada neurônio
    #neuron_class_map = {neuron: Counter(classes).most_common(1)[0][0] for neuron, classes in neuron_classes.items()}
    
    # Cria um dicionário vazio para armazenar o resultado
    neuron_class_map = {}
    
    # Percorre todos os neurônios e suas listas de classes
    for neuron, classes in neuron_classes.items():
        
        # Conta quantas vezes cada classe aparece
        contador = Counter(classes)
            
        # Obtém a classe mais comum (retorna uma lista de tuplas: [(classe, contagem)])
        mais_comum = contador.most_common(1)
           
        # Pega apenas o valor da classe (primeiro elemento da tupla)
        classe_majoritaria = mais_comum[0][0]
    
        # Associa essa classe ao neurônio no novo dicionário
        neuron_class_map[neuron] = classe_majoritaria
    
    # ============================================================
    # PREDIÇÕES PARA TODAS AS AMOSTRAS
    # ============================================================
    y_pred = np.array([neuron_class_map[som.winner(sample)] for sample in data_scaled])
    
    # ============================================================
    # MATRIZ DE CONFUSÃO
    # ============================================================
    cm = confusion_matrix(labels, y_pred)
    if plot_cm:

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fundo (0)', 'Estrada (1)'])
        disp.plot(cmap='Blues')
        plt.title("Matriz de Confusão - SOM")
        plt.show()
    
    # ============================================================
    # MÉTRICAS DE DESEMPENHO
    # ============================================================
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)

    return accuracy, precision, recall, f1, cm