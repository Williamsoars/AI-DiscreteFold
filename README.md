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
git clone https://github.com/Williamsoars/AI-DiscreteFold.git
cd AI-DiscreteFold
pip install -r requirements.txt

# Configurar HHblits (necessário para MSA)
export HHLIB=/caminho/para/hh-suite
```
### Uso básico
```
from pipeline import run_pipeline

# Pipeline completo para uma proteína
results = run_pipeline(
    uniprot_id="P69905",  # Hemoglobina
    db_path="/path/to/uniclust30",
    epochs=50
)
```
### Exemplo de saída
```
✅ Pipeline concluído!

📊 Métricas:
- RMSD: 4.23 Å
- Tempo de execução: 12m 35s
- Resíduos alinhados: 142/146
```
### Estrutura do projeto
```
/protein-folding-toolkit
├── data/                  # Dados brutos e processados
├── models/                # Modelos treinados
├── src/
│   ├── data/              # Obtenção e preparação de dados
│   ├── features/          # Extração de features
│   ├── models/            # Arquiteturas de ML
│   ├── evaluation/        # Métricas e visualização
│   └── utils/             # Funções auxiliares
├── notebooks/             # Exemplos e experimentos
└── docs/                  # Documentação
```
### Requisitos
```
Python 3.8+

PyTorch 1.10+

BioPython

ESM (Facebook Research)

HH-suite (para MSA)

Matplotlib/Plotly

