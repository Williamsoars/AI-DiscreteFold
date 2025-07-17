import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_structure(coords: np.ndarray, title="Estrutura da Proteína", color='blue'):
    """
    Plota coordenadas 3D (ex: átomos CA) em um gráfico interativo.

    Args:
        coords (np.ndarray): Coordenadas (N x 3)
        title (str): Título do gráfico
        color (str): Cor dos pontos/linhas
    """
    if coords.shape[1] != 3:
        raise ValueError("A entrada deve ter shape (N, 3)")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.plot(x, y, z, marker='o', color=color, linewidth=1, markersize=3)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)
    plt.tight_layout()
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    coords = np.cumsum(np.random.randn(100, 3), axis=0)  # Trajetória simulada
    plot_3d_structure(coords, title="Exemplo de Proteína Simulada")
