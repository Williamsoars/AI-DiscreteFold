from Bio.PDB import PDBList, PDBParser
import numpy as np
from pathlib import Path

def download_pdb(pdb_id: str, save_dir="pdb_structures") -> str:
    """
    Baixa um arquivo PDB da RCSB PDB.

    Args:
        pdb_id (str): ID do PDB (ex: "1A3N")
        save_dir (str): Diretório onde salvar o arquivo

    Returns:
        str: Caminho do arquivo PDB salvo
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=save_dir, file_format="pdb")
    return f"{save_dir}/pdb{pdb_id.lower()}.ent"

def extract_ca_coordinates(pdb_file: str) -> np.ndarray:
    """
    Extrai as coordenadas 3D dos átomos CA da estrutura da proteína.

    Args:
        pdb_file (str): Caminho para o arquivo PDB

    Returns:
        np.ndarray: Coordenadas dos átomos CA (N x 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_coord())
        break  # Usa apenas o primeiro modelo

    return np.array(coords)

