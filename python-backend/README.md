# AI Music Detector Backend

Backend en Python para detectar si una canción fue generada con Inteligencia Artificial.

## Características

- **Extracción de metadatos** con `mutagen`.
- **Identificación de canciones** con `ShazamIO`.
- **Búsqueda en YouTube** con `youtube-search-python` para detectar video oficial y covers.
- **Detección de firmas de herramientas de IA** en metadatos (Suno, Udio, ElevenLabs, etc.).
- **Análisis acústico** con `librosa` y `pyAudioAnalysis`:
  - Zero Crossing Rate
  - Spectral Centroid, Bandwidth, Contrast, Rolloff, Flux
  - MFCCs
  - RMSE
- **Análisis de letra** mediante heurísticas (naturalidad, originalidad, coherencia emocional, repetición, especificidad narrativa, variación estructural).
- **Inferencia difusa** Mamdani-style para combinar evidencias.
- **API REST** con `FastAPI` lista para consumir desde React Native o cualquier cliente.
- **Dockerizado** para despliegue en la nube.

## Estructura

```
python-backend/
├── app/
│   ├── main.py              # FastAPI app
│   ├── analyzer.py          # Orquestador del análisis
│   ├── config.py            # Constantes y firmas de IA
│   ├── models.py            # Pydantic models
│   ├── utils.py             # Helpers
│   └── services/
│       ├── metadata.py      # Metadatos + firmas IA
│       ├── shazam_service.py
│       ├── youtube_service.py
│       ├── acoustic.py      # librosa / pyAudioAnalysis
│       └── lyrics.py        # Análisis de letra
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Uso local

### 1. Crear entorno virtual

```bash
cd python-backend
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Nota: `pyAudioAnalysis` puede requerir `ffmpeg`. En Linux/macOS instálalo con tu gestor de paquetes. En Windows usa Chocolatey o descarga desde el sitio oficial.

### 3. Ejecutar servidor

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Probar healthcheck

```bash
curl http://localhost:8000/api/health
```

### 5. Analizar una canción

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "audio=@/ruta/a/tu/cancion.mp3" \
  -F "lyrics=Opcional: letra de la canción"
```

## Uso con Docker

### Construir y ejecutar

```bash
cd python-backend
docker-compose up --build -d
```

### Ver logs

```bash
docker-compose logs -f
```

### Detener

```bash
docker-compose down
```

## Despliegue en la nube

El contenedor expone el puerto `8000`. Puedes desplegarlo en:

- **Google Cloud Run**: construye la imagen, súbela a Artifact Registry y despliega un servicio.
- **AWS ECS / Fargate** o **App Runner**.
- **Azure Container Apps**.
- **Railway / Render / Fly.io**.

Ejemplo de build y push manual:

```bash
docker build -t tu-registry/ai-music-detector:latest .
docker push tu-registry/ai-music-detector:latest
```

## Variables de entorno

| Variable | Default | Descripción |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Host de escucha |
| `PORT` | `8000` | Puerto |
| `MAX_FILE_BYTES` | `104857600` | Tamaño máximo de archivo (100 MB) |
| `ANALYSIS_SEGMENT_SECONDS` | `60` | Segundos de audio a analizar por segmento |
| `ANALYSIS_HOP_SECONDS` | `30` | Salto entre segmentos |
| `SHAZAM_TIMEOUT` | `45` | Timeout para ShazamIO |
| `YOUTUBE_SEARCH_TIMEOUT` | `20` | Timeout para búsquedas de YouTube |
| `HUMAN_THRESHOLD` | `35.0` | Umbral para clasificar como humano |
| `AI_THRESHOLD` | `65.0` | Umbral para clasificar como IA |

## Respuesta JSON

La API devuelve un JSON con:

- `audio_file`: nombre del archivo.
- `scale`: escala 0 = Humano, 100 = Generado por IA.
- `ai_score`: puntaje final 0-100.
- `ai_score_bar`: barra ASCII visual.
- `classification`: Humano / Generado por IA / Indeterminado.
- `decision_layer` y `decision_reasoning`: capa de decisión y justificación.
- `shazam_analysis`: resultado de ShazamIO.
- `metadata_analysis`: metadatos y detección de firmas IA.
- `acoustic_and_lyrics_analysis`: features acústicas, letra e inferencia difusa.
- `youtube_analysis`: resultados de YouTube (oficial/cover).
- Resúmenes de `Zero Crossing Rate`, `Spectral Centroid`, etc.

## Notas importantes

- El análisis de letra se basa en la letra embebida en los metadatos o en el texto opcional enviado a la API.
- Si `ShazamIO` o `youtube-search-python` fallan (límites de red, cambios de API), el análisis continúa con las capas restantes y reporta el error en los campos correspondientes.
- `pyAudioAnalysis` se usa opcionalmente; si no está disponible, el análisis recae completamente en `librosa`.
