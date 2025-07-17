import os
import requests
import gzip
import shutil
import csv
import json
from pathlib import Path
from Bio import SeqIO
from Bio.PDB import PDBList

def download_swissprot_fasta(out_path="dataset/swissprot.fasta"):
    """Baixa o FASTA completo do SwissProt."""
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    out_path_gz = out_path + ".gz"
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    
    print("Baixando SwissProt...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path_gz, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print("Descompactando...")
    with gzip.open(out_path_gz, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(out_path_gz)
    print(f"SwissProt salvo em: {out_path}")
    return out_path

def filter_fasta_by_length_and_redundancy(fasta_path: str, min_len=50, max_len=700) -> str:
    """
    Filtra FASTA por tamanho e remove sequências redundantes (idênticas).
    """
    filtered_path = fasta_path.replace(".fasta", f"_filtered_{min_len}_{max_len}.fasta")
    seen = set()
    filtered = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        if min_len <= len(seq) <= max_len and seq not in seen:
            seen.add(seq)
            filtered.append(record)

    SeqIO.write(filtered, filtered_path, "fasta")
    print(f"{len(filtered)} sequências filtradas salvas em: {filtered_path}")
    return filtered_path

def generate_metadata(fasta_path: str, out_csv="dataset/metadata.csv", out_json="dataset/metadata.json"):
    """Cria metadados de um FASTA (ID, descrição, tamanho)."""
    metadata = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        metadata.append({
            "id": record.id,
            "description": record.description,
            "length": len(record.seq)
        })

    with open(out_csv, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["id", "description", "length"])
        writer.writeheader()
        writer.writerows(metadata)

    with open(out_json, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadados salvos em: {out_csv} e {out_json}")
    return metadata

def download_pdb_structure(pdb_id: str, out_dir="dataset/pdb_structures") -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir=out_dir, file_format="pdb")
    return f"{out_dir}/pdb{pdb_id.lower()}.ent"

# Exemplo de uso do pipeline completo
if __name__ == "__main__":
    fasta_path = download_swissprot_fasta()
    filtered_fasta = filter_fasta_by_length_and_redundancy(fasta_path)
    generate_metadata(filtered_fasta)
    # download_pdb_structure("1A3N")  # Exemplo opcional
