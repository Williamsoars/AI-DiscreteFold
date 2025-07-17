import os
import subprocess
from pathlib import Path

def run_hhblits(fasta_path: str, output_dir: str, db_path: str):
    """
    Executa HHblits para gerar o MSA a partir de uma sequência FASTA.
    
    Args:
        fasta_path (str): Caminho para o arquivo FASTA.
        output_dir (str): Diretório onde o MSA será salvo.
        db_path (str): Caminho para o banco de dados (ex: uniclust30).
    
    Returns:
        str: Caminho do arquivo .a3m gerado.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_a3m = os.path.join(output_dir, "output.a3m")

    command = [
        "hhblits",
        "-i", fasta_path,
        "-o", os.path.join(output_dir, "output.hhr"),
        "-oa3m", output_a3m,
        "-d", db_path,
        "-n", "3",
        "-e", "1e-3",
        "-cpu", "4"
    ]

    print(f"Executando HHblits: {' '.join(command)}")
    subprocess.run(command, check=True)
    return output_a3m


