import os
from uniprot_to_fasta import get_fasta_from_uniprot
from msa_generator import run_hhblits
from msa_embedding import esm_embedding_from_fasta
from pdb_structure import download_pdb, extract_ca_coordinates
from model_trainer import train_folding_model, FoldingRegressor
from structure_eval import compute_rmsd
import torch

def run_pipeline(uniprot_id: str, pdb_id: str, db_path: str, epochs=20):
    # 1. Baixar FASTA UniProt
    fasta_path = get_fasta_from_uniprot(uniprot_id)
    print(f"FASTA salvo em: {fasta_path}")

    # 2. Gerar MSA (opcional, pois ESM usa sequência direta, mas mantemos para flexibilidade)
    msa_dir = "msa_output"
    msa_path = run_hhblits(fasta_path, msa_dir, db_path)
    print(f"MSA gerado em: {msa_path}")

    # 3. Gerar embeddings ESM-1b da sequência
    embeddings = esm_embedding_from_fasta(fasta_path)
    print(f"Embedding ESM shape: {embeddings.shape}")

    # 4. Baixar estrutura PDB e extrair coordenadas CA
    pdb_path = download_pdb(pdb_id)
    true_coords = extract_ca_coordinates(pdb_path)
    print(f"Coordenadas CA extraídas, shape: {true_coords.shape}")

    # 5. Treinar modelo
    # Aqui o treino espera fasta e pdb, mas já temos embeddings, então adaptamos para passar embeddings e coords
    # Para protótipo, reusamos train_folding_model que usa fasta, mas poderia ser otimizado
    model = train_folding_model(fasta_path, pdb_path, epochs=epochs)

    # 6. Avaliar modelo (inferência simples)
    model.eval()
    with torch.no_grad():
        pred_coords = model(embeddings.float()).cpu().numpy()

    rmsd = compute_rmsd(true_coords, pred_coords)
    print(f"RMSD da predição: {rmsd:.4f}")


