import os
import torch
from uniprot_to_fasta import get_fasta_from_uniprot
from msa_generator import run_hhblits
from msa_embedding import esm_embedding_from_fasta
from pdb_structure import download_pdb, extract_ca_coordinates
from model_trainer import train_folding_model
from structure_eval import compute_rmsd
from visualize_structure import plot_3d_structure
from uniprot_to_pdb_mapping import map_uniprot_to_pdb


def run_pipeline(uniprot_id: str, db_path: str, epochs=20):
    # 1. Verifica se existe estrutura PDB associada ao UniProt
    pdb_ids = map_uniprot_to_pdb(uniprot_id)
    if not pdb_ids:
        print(f"Nenhuma estrutura PDB encontrada para {uniprot_id}.")
        return
    pdb_id = pdb_ids[0]  # Usar o primeiro mapeado

    # 2. Baixar FASTA UniProt
    fasta_path = get_fasta_from_uniprot(uniprot_id)
    print(f"FASTA salvo em: {fasta_path}")

    # 3. Gerar MSA (opcional)
    msa_dir = "msa_output"
    msa_path = run_hhblits(fasta_path, msa_dir, db_path)
    print(f"MSA gerado em: {msa_path}")

    # 4. Gerar embeddings ESM-1b
    embeddings = esm_embedding_from_fasta(fasta_path)
    print(f"Embedding ESM shape: {embeddings.shape}")

    # 5. Baixar estrutura PDB e extrair coordenadas CA
    pdb_path = download_pdb(pdb_id)
    true_coords = extract_ca_coordinates(pdb_path)
    print(f"Coordenadas CA extraídas, shape: {true_coords.shape}")

    # 6. Treinar modelo
    model = train_folding_model(fasta_path, pdb_path, epochs=epochs)

    # 7. Inferência
    model.eval()
    with torch.no_grad():
        pred_coords = model(embeddings.float()).cpu().numpy()

    # 8. Avaliação
    rmsd = compute_rmsd(true_coords, pred_coords)
    print(f"RMSD da predição: {rmsd:.4f}")

    # 9. Visualização
    print("Visualizando estrutura real:")
    plot_3d_structure(true_coords, title=f"PDB real: {pdb_id}", color='green')
    print("Visualizando estrutura predita:")
    plot_3d_structure(pred_coords, title="Estrutura predita", color='orange')


if __name__ == "__main__":
    uniprot_id = "P69905"  # Exemplo: hemoglobina
    db_path = "/caminho/para/uniclust30_2021_03"  # Ajustar conforme seu ambiente
    run_pipeline(uniprot_id, db_path)



