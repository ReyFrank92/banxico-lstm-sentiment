
# Banxico Minutes Sentiment — LSTM (Hawkish/Dovish/Neutral)

Repositorio anexo del ensayo. Implementa una red LSTM para clasificar el tono monetario de minutas del Banco de México.

## Contenido
- `banxico_lstm_sentiment_clean.ipynb`: cuaderno principal (sin comentarios en el código).
- `src_main_clean.py`: script Python exportado del cuaderno.
- `.gitignore`: exclusiones estándar.
- `requirements.txt`: dependencias estimadas.
- `LICENSE`: licencia MIT.
- Carpetas vacías: `data/`, `models/`, `reports/`.

## Ejecución rápida
```bash
pip install -r requirements.txt
jupyter lab banxico_lstm_sentiment_clean.ipynb
# o ejecutar directamente:
python src_main_clean.py
```

## Esquema de etiquetas
- `hawkish` → 0
- `dovish`  → 1
- `neutral` → 2
