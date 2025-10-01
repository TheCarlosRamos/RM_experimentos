import numpy as np
import matplotlib.pyplot as plt

def extract_radial_trajectories(image, n_radial_lines=64):
    k_space_full = np.fft.fft2(image)
    k_space_shifted = np.fft.fftshift(k_space_full)
    
    N, M = k_space_shifted.shape
    center_x, center_y = N//2, M//2
    
    todos_valores = []
    todas_coordenadas = []
    
    angles = np.linspace(0, np.pi, n_radial_lines, endpoint=False)
    max_radius = min(center_x, center_y)
    
    for angle in angles:
        r = np.linspace(0, max_radius, max_radius)
        kx_line = r * np.cos(angle)
        ky_line = r * np.sin(angle)
        
        idx_x = np.round(center_x + kx_line).astype(int)
        idx_y = np.round(center_y + ky_line).astype(int)
        
        valid = (idx_x >= 0) & (idx_x < N) & (idx_y >= 0) & (idx_y < M)
        idx_x = idx_x[valid]
        idx_y = idx_y[valid]
        
        if len(idx_x) > 0:
            line_values = k_space_shifted[idx_x, idx_y]
            line_coords = np.column_stack([kx_line[valid], ky_line[valid]])
            
            todos_valores.extend(line_values)
            todas_coordenadas.extend(line_coords)
    
    valores_dft = np.array(todos_valores)
    coordenadas = np.array(todas_coordenadas)
    
    return valores_dft, coordenadas

def plot_radial_trajectories(image, valores_dft, coordenadas):
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
    
    n_radial_lines = 10
    points_per_line = len(coordenadas) // n_radial_lines
    
    for i in range(min(16, n_radial_lines)):
        start_idx = i * points_per_line
        end_idx = start_idx + points_per_line
        if end_idx <= len(coordenadas):
            coords_slice = coordenadas[start_idx:end_idx]
            axes[2].plot(coords_slice[:, 1] + M//2, 
                        coords_slice[:, 0] + N//2, 
                        'r-', alpha=0.5, linewidth=0.8)
    
    axes[2].set_title(f'Trajetória Radial ({n_radial_lines} linhas)')
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
    
    n_radial_lines = 200
    valores_dft, coordenadas = extract_radial_trajectories(x, n_radial_lines)
    
    print(f"Numero de linhas radiais: {n_radial_lines}")
    print(f"Valores DFT shape: {valores_dft.shape}")
    print(f"Coordenadas shape: {coordenadas.shape}")
    print(f"Total de pontos amostrados: {len(valores_dft)}")
    
    plot_radial_trajectories(x, valores_dft, coordenadas)