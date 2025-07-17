import requests
import csv

def map_uniprot_to_pdb(uniprot_id: str) -> list[str]:
    """
    Mapeia um UniProt ID para IDs PDB usando a API do PDBe.
    """
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot_id.lower()}"
    r = requests.get(url)
    if r.status_code != 200:
        raise Exception(f"Erro ao acessar PDBe para {uniprot_id}")

    data = r.json()
    if uniprot_id not in data:
        return []

    pdbs = [entry['pdb_id'] for entry in data[uniprot_id]]
    return pdbs

def map_multiple_uniprots(uniprot_ids: list[str], out_csv="dataset/uniprot_pdb_map.csv"):
    """
    Mapeia uma lista de UniProt IDs para PDBs e salva como CSV.
    """
    with open(out_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["uniprot_id", "pdb_ids"])
        for uid in uniprot_ids:
            try:
                pdbs = map_uniprot_to_pdb(uid)
                writer.writerow([uid, ";".join(pdbs)])
                print(f"{uid} → {pdbs}")
            except Exception as e:
                print(f"Erro com {uid}: {e}")

# Exemplo de uso
if __name__ == "__main__":
    uniprots = ["P69905", "P68871", "Q9Y263"]
    map_multiple_uniprots(uniprots)
