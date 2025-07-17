import numpy as np
import torch
import esm
from Bio import SeqIO

def parse_a3m(file_path: str) -> list[str]:
    """
    Extrai as sequências de um arquivo .a3m.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        sequences = [line.strip() for line in lines if not line.startswith('>')]
    return sequences

def encode_msa_onehot(msa: list[str], alphabet="ACDEFGHIKLMNPQRSTVWY-") -> np.ndarray:
    """
    Converte MSA em codificação one-hot simplificada (para protótipo).
    """
    alphabet_dict = {char: i for i, char in enumerate(alphabet)}
    encoded = np.zeros((len(msa), len(msa[0]), len(alphabet)), dtype=np.float32)

    for i, seq in enumerate(msa):
        for j, char in enumerate(seq):
            if char in alphabet_dict:
                encoded[i, j, alphabet_dict[char]] = 1.0
    return encoded

def msa_to_embedding(a3m_path: str) -> np.ndarray:
    msa = parse_a3m(a3m_path)
    embedding = encode_msa_onehot(msa)
    return embedding

def esm_embedding_from_fasta(fasta_path: str) -> torch.Tensor:
    """
    Gera embeddings do ESM-1b para a sequência FASTA.
    """
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    records = list(SeqIO.parse(fasta_path, "fasta"))
    data = [(record.id, str(record.seq)) for record in records]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]

    # Remove CLS/SEP tokens, pega só a sequência
    sequence_representations = []
    for i, (_, seq) in enumerate(data):
        sequence_representations.append(token_representations[i, 1:len(seq)+1].mean(0))

    return torch.stack(sequence_representations)

def main():
    # Exemplo: one-hot embedding
    embedding = msa_to_embedding("msa_output/output.a3m")
    print("Shape do embedding MSA (one-hot):", embedding.shape)

    # Exemplo: ESM-1b embedding
    esm_emb = esm_embedding_from_fasta("example.fasta")
    print("Shape do embedding ESM-1b:", esm_emb.shape)

if __name__ == "__main__":
    main()
