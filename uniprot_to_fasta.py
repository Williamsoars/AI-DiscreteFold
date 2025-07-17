import requests
import time

def get_uniprot_entry(protein_id):
    """
    Busca entrada do UniProt para um ID de proteína.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Erro {response.status_code} ao buscar {protein_id}")
        return None

def extract_fasta_from_uniprot(protein_id):
    """
    Extrai sequência FASTA a partir do UniProt.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Erro ao buscar FASTA de {protein_id}")
        return None


