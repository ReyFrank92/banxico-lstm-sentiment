# Banxico Minutes Sentiment — LSTM (Hawkish/Dovish/Neutral)

Repositorio anexo del ensayo. Implementa una red LSTM para clasificar el tono monetario de minutas del Banco de México.

## Contenido
- `banxico_lstm_sentiment_clean.ipynb`: cuaderno principal (sin comentarios en el código).
- `src_main_clean.py`: script Python exportado del cuaderno.
- `.gitignore`: exclusiones estándar.
- `requirements.txt`: dependencias estimadas.
- `LICENSE`: licencia MIT.
- Carpetas vacías: `data/`, `models/`, `reports/`.

##  Datos

Este proyecto utiliza como insumo las minutas del Banco de México, que forman parte del repositorio público **WorldCentralBanks** del *Georgia Institute of Technology Fintech Lab*.

- Repositorio principal: [gtfintechlab/WorldCentralBanks](https://github.com/gtfintechlab/WorldCentralBanks)  
- Archivo específico para Banco de México:  
  [`final_data.csv`](https://github.com/gtfintechlab/WorldCentralBanks/blob/main/final_data/bank_of_mexico/final_data.csv)

###  Instrucciones para reproducir
1. Descargue el archivo [`final_data.csv`](https://github.com/gtfintechlab/WorldCentralBanks/raw/main/final_data/bank_of_mexico/final_data.csv).  
2. Guárdelo en la carpeta `data/` de este repositorio.  
3. Ejecute el cuaderno `banxico_lstm_sentiment_clean.ipynb`.  

Ejemplo de carga en Python:
```python
from pathlib import Path
import pandas as pd

# Ruta relativa dentro del repo
csv_path = Path("data/final_data.csv")
df = pd.read_csv(csv_path)

print(df.head())
