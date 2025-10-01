import numpy as np
import matplotlib.pyplot as plt

def extract_rectangular_trajectories(image, n_lines=64):
    k_space_full = np.fft.fft2(image)
    k_space_shifted = np.fft.fftshift(k_space_full)
    
    N, M = k_space_shifted.shape
    
    todos_valores = []
    todas_coordenadas = []
    
    step = max(1, N // n_lines)
    line_indices = range(0, N, step)
    
    kx_coords = np.linspace(-1, 1, M)
    ky_coords = np.linspace(-1, 1, N)
    
    for i in line_indices:
        line_values = k_space_shifted[i, :]
        
        for j in range(M):
            coord_x = kx_coords[j]
            coord_y = ky_coords[i]
            todas_coordenadas.append([coord_x, coord_y])
        
        todos_valores.extend(line_values)
    
    valores_dft = np.array(todos_valores)
    coordenadas = np.array(todas_coordenadas)
    
    return valores_dft, coordenadas

def plot_rectangular_trajectories(image, valores_dft, coordenadas):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagem Original')
    axes[0].axis('off')
    
    k_space_full = np.fft.fft2(image)
    k_space_shifted = np.fft.fftshift(k_space_full)
    N, M = k_space_shifted.shape
    
    axes[1].imshow(np.log10(np.abs(k_space_shifted) + 1), cmap='gray')
    axes[1].set_title('Espaço-k Completo')
    axes[1].axis('off')
    
    axes[2].imshow(np.log10(np.abs(k_space_shifted) + 1), cmap='gray')
    
    n_lines = len(np.unique(coordenadas[:,1]))
    
    for i in range(min(16, n_lines)):
        start_idx = i * M
        end_idx = start_idx + M
        if end_idx <= len(coordenadas):
            coords_slice = coordenadas[start_idx:end_idx]
            axes[2].plot(coords_slice[:, 0] * M//2 + M//2, 
                        coords_slice[:, 1] * N//2 + N//2, 
                        'g-', alpha=0.5, linewidth=0.8)
    
    axes[2].set_title(f'Trajetória Retangular ({n_lines} linhas)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        x = plt.imread('exemplo_corte_axial.jpg')
        if len(x.shape) == 3:
            x = x[:, :, 0]
        print(f"Imagem carregada: {x.shape}")
    except FileNotFoundError:
        print("Arquivo não encontrado. Criando imagem de exemplo...")
        x = np.random.randn(256, 256)
        x = np.abs(np.fft.ifft2(np.fft.fft2(x) * np.exp(-0.001*(np.arange(256)[:, None]**2 + np.arange(256)[None, :]**2)))).real
        print(f"Imagem exemplo criada: {x.shape}")
    
    n_lines = 90
    valores_dft, coordenadas = extract_rectangular_trajectories(x, n_lines)
    
    print(f"Numero de linhas retangulares: {n_lines}")
    print(f"Valores DFT shape: {valores_dft.shape}")
    print(f"Coordenadas shape: {coordenadas.shape}")
    print(f"Total de pontos amostrados: {len(valores_dft)}")
    
    plot_rectangular_trajectories(x, valores_dft, coordenadas)