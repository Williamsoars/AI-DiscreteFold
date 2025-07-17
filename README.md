# 🧬 DiscreteFold

Biblioteca Python para predição da estrutura 3D de proteínas a partir de sequências de aminoácidos.

## ✨ Funcionalidades

- **Obtenção de dados**:
  - Download automático de sequências (UniProt)
  - Mapeamento para estruturas PDB
  - Geração de MSA com HHblits

- **Processamento**:
  - Extração de coordenadas CA de arquivos PDB
  - Geração de embeddings (ESM-1b, one-hot)
  - Pré-processamento de sequências

- **Modelagem**:
  - Modelo neural para predição de coordenadas 3D
  - Treinamento com loss RMSD
  - Suporte a GPU/CUDA

- **Avaliação**:
  - Cálculo de RMSD
  - Visualização 3D interativa
  - Comparação estrutura real vs predita

## 🚀 Instalação

```bash
git clone https://github.com/seu-usuario/protein-folding-toolkit.git
cd protein-folding-toolkit
pip install -r requirements.txt

# Configurar HHblits (necessário para MSA)
export HHLIB=/caminho/para/hh-suite
