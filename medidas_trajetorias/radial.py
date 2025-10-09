import numpy as np
import matplotlib.pyplot as plt

def calculate_radial_coordinates(image_shape, n_radial_lines=64):
    N, M = image_shape
    center_x, center_y = N // 2, M // 2
    todas_coordenadas = []
    angles = np.linspace(0, 2 * np.pi, n_radial_lines, endpoint=False)
    max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
    for angle in angles:
        r = np.linspace(0, max_radius, int(max_radius))
        kx_line = r * np.cos(angle)
        ky_line = r * np.sin(angle)
        line_coords = np.column_stack([kx_line, ky_line])
        todas_coordenadas.extend(line_coords)
    coordenadas = np.array(todas_coordenadas)
    return coordenadas

def extract_measurements(image, coordinates):
    k_space_full = np.fft.fft2(image)
    N, M = image.shape
    center_x, center_y = N // 2, M // 2
    todos_valores = []
    coordenadas_validas = []
    for coord in coordinates:
        kx, ky = coord
        idx_x = int(np.round(center_x + kx))
        idx_y = int(np.round(center_y + ky))
        if 0 <= idx_x < N and 0 <= idx_y < M:
            value = k_space_full[idx_x, idx_y]
            todos_valores.append(value)
            coordenadas_validas.append([kx, ky])
    valores_dft = np.array(todos_valores)
    coordenadas_validas = np.array(coordenadas_validas)
    return valores_dft, coordenadas_validas

def extract_radial_trajectories(image, n_radial_lines=64):
    coordenadas = calculate_radial_coordinates(image.shape, n_radial_lines)
    valores_dft, coordenadas_validas = extract_measurements(image, coordenadas)
    return valores_dft, coordenadas_validas

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
    k_space_mag = np.log10(np.abs(k_space_shifted) + 1)
    axes[2].imshow(k_space_mag, cmap='gray')
    n_radial_lines = len(np.unique(np.arctan2(coordenadas[:, 1], coordenadas[:, 0])))
    points_per_line = len(coordenadas) // n_radial_lines
    for i in range(n_radial_lines):
        start_idx = i * points_per_line
        end_idx = start_idx + points_per_line
        if end_idx <= len(coordenadas):
            coords_slice = coordenadas[start_idx:end_idx]
            axes[2].plot(coords_slice[:, 1] + M // 2,
                         coords_slice[:, 0] + N // 2,
                         'r-', alpha=0.8, linewidth=1.0)
    axes[2].set_title(f'Trajetória Radial ({n_radial_lines} linhas)')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        x = plt.imread('exemplo_corte_axial.jpg')
        if len(x.shape) == 3:
            x = x[:, :, 0]
    except FileNotFoundError:
        x = np.random.randn(512, 512)
        x = np.abs(np.fft.ifft2(np.fft.fft2(x) * np.exp(-0.001*(np.arange(512)[:, None]**2 + np.arange(512)[None, :]**2)))).real
    n_radial_lines = 14
    valores_dft, coordenadas = extract_radial_trajectories(x, n_radial_lines)
    plot_radial_trajectories(x, valores_dft, coordenadas)
