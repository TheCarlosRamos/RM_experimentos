import numpy as np
import matplotlib.pyplot as plt

def extract_spiral_trajectories(image, n_turns=8):
    k_space_full = np.fft.fft2(image)
    k_space_shifted = np.fft.fftshift(k_space_full)
    
    N, M = k_space_shifted.shape
    center_x, center_y = N//2, M//2
    
    todos_valores = []
    todas_coordenadas = []
    
    theta_max = 2 * np.pi * n_turns
    n_points = 1000
    theta = np.linspace(0, theta_max, n_points)
    
    r_max = min(center_x, center_y)
    a = r_max / theta_max
    r = a * theta
    
    kx_spiral = r * np.cos(theta)
    ky_spiral = r * np.sin(theta)
    
    idx_x = np.round(center_x + kx_spiral).astype(int)
    idx_y = np.round(center_y + ky_spiral).astype(int)
    
    valid = (idx_x >= 0) & (idx_x < N) & (idx_y >= 0) & (idx_y < M)
    idx_x = idx_x[valid]
    idx_y = idx_y[valid]
    kx_spiral = kx_spiral[valid]
    ky_spiral = ky_spiral[valid]
    
    spiral_values = k_space_shifted[idx_x, idx_y]
    spiral_coords = np.column_stack([kx_spiral, ky_spiral])
    
    valores_dft = spiral_values
    coordenadas = spiral_coords
    
    return valores_dft, coordenadas

def plot_spiral_trajectories(image, valores_dft, coordenadas):
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
    
    axes[2].plot(coordenadas[:, 1] * M//2 + M//2, 
                coordenadas[:, 0] * N//2 + N//2, 
                'b-', alpha=0.8, linewidth=1.5)
    
    axes[2].set_title('Trajetória Espiral')
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
    
    n_turns = 82
    valores_dft, coordenadas = extract_spiral_trajectories(x, n_turns)
    
    print(f"Numero de voltas da espiral: {n_turns}")
    print(f"Valores DFT shape: {valores_dft.shape}")
    print(f"Coordenadas shape: {coordenadas.shape}")
    print(f"Total de pontos amostrados: {len(valores_dft)}")
    
    plot_spiral_trajectories(x, valores_dft, coordenadas)