import numpy as np
from scipy.spatial.distance import euclidean

def compute_rmsd(true_coords: np.ndarray, pred_coords: np.ndarray) -> float:
    """
    Calcula o RMSD (Root Mean Square Deviation) entre duas estruturas.

    Args:
        true_coords (np.ndarray): Coordenadas reais (N x 3)
        pred_coords (np.ndarray): Coordenadas preditas (N x 3)

    Returns:
        float: RMSD entre as estruturas
    """
    if true_coords.shape != pred_coords.shape:
        raise ValueError(f"Shapes incompatíveis: {true_coords.shape} vs {pred_coords.shape}")

    diff = true_coords - pred_coords
    rmsd = np.sqrt((diff ** 2).sum() / len(true_coords))
    return rmsd


