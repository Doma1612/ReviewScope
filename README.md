# ReviewScope

ReviewScope groups large text collections into topics. It splits each document into sentences, embeds them, clusters them, and labels every cluster with a local LLM. The web app shows the clusters on an interactive map, lets users correct them (rename, merge, move, mark as junk), and answers questions about a project through a chat grounded in the documents.

Any CSV or JSONL file with a text column works. Two datasets are included in [`datasets/`](datasets/).

## Run it locally

Requirements: Docker Desktop (Windows, macOS) or Docker Engine with the Compose plugin (Linux), 8 GB of memory for Docker, and about 25 GB of free disk space.

1. Create an empty folder and download the configuration files and the small demo dataset into it:

   ```
   curl -LO https://raw.githubusercontent.com/Doma1612/ReviewScope/main/docker-compose.release.yml
   curl -L -o .env https://raw.githubusercontent.com/Doma1612/ReviewScope/main/.env.example
   curl -LO https://raw.githubusercontent.com/Doma1612/ReviewScope/main/datasets/hotels_100.jsonl
   ```

   On Windows PowerShell, write `curl.exe` instead of `curl`.

2. Open `.env` and replace the values of `SECRET_KEY` and `POSTGRES_PASSWORD` with your own random strings.

3. Start the app:

   ```
   docker compose -f docker-compose.release.yml up -d
   ```

   The first start downloads the images (about 7 GB) and the language models (about 5 GB). Check the progress with `docker compose -f docker-compose.release.yml ps`. The app is ready when `ollama` reports `healthy`.

4. Open http://localhost:5173 and register an account. Use exactly this address; `127.0.0.1` is rejected by the API.

5. Click **New Project**, upload `hotels_100.jsonl`, confirm the detected columns, and start. The project card shows each pipeline step. After a few minutes the project is **Ready** and **Open** shows the clusters.

To stop the app, run `docker compose -f docker-compose.release.yml down`. Adding `-v --rmi all` also deletes all data, models and images.

The images run on amd64 and arm64 (Apple Silicon). `REVIEWSCOPE_VERSION` in `.env` selects a release, for example `0.1.0`; the default `latest` is the newest stable release.

## Datasets

| File | Content | Size | Recommended hardware |
|---|---|---|---|
| `hotels_100.jsonl` | 100 hotel reviews | 0.1 MB | Any laptop, a few minutes |
| `hotels_50k.jsonl` | 50,000 hotel reviews | 48 MB | GPU server, about 1 hour |
| `europarl_en_26k.jsonl` | 26,391 English speeches from the European Parliament | 53 MB | GPU server |

Sources, licences and how to rebuild the files are described in [`datasets/README.md`](datasets/README.md).

## Large datasets: use a GPU server

The pipeline works on CPU, but embedding and sentiment scoring are much faster on an NVIDIA GPU, and the 50k dataset produces about 450,000 sentences. If a GPU server is available, run the app there and open it on your laptop through an SSH tunnel:

1. On the server, clone the repository and create the `.env`:

   ```
   git clone https://github.com/Doma1612/ReviewScope.git && cd ReviewScope
   cp .env.example .env
   ```

   Set `SECRET_KEY` and `POSTGRES_PASSWORD` in `.env`.

2. Start the stack with the GPU override that fits the server:

   | Server | Command |
   |---|---|
   | NVIDIA driver and nvidia-container-toolkit installed | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build` |
   | ltgpu1 (old Docker, no toolkit) | `docker-compose -f docker-compose.yml -f docker-compose.ltgpu.yml up -d --build` |

   On a shared server each pipeline run claims one idle GPU and falls back to CPU when all are busy. `ML_MAX_GPUS` in `.env` changes the limit.

3. On your laptop, open the tunnel and keep it running:

   ```
   ssh -L 5173:localhost:5173 -L 8000:localhost:8000 <user>@<server>
   ```

4. Open http://localhost:5173 on your laptop.

## Development from source

```
git clone https://github.com/Doma1612/ReviewScope.git && cd ReviewScope
cp .env.example .env
docker compose up -d --build
```

This builds the images locally and mounts `src/backend` and `src/frontend` into the containers, so code changes reload without a rebuild. Changes in `src/reviewscope_ml` need `docker compose build worker api`. The API documentation is at http://localhost:8000/docs.

The ML research tooling (notebooks, model sweeps, evaluation reports) runs outside Docker:

```
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest tests
```

## Repository layout

| Path | Content |
|---|---|
| [`src/backend`](src/backend/README.md) | FastAPI API, Celery worker, database models and migrations |
| [`src/frontend`](src/frontend/README.md) | React web app |
| [`src/reviewscope_ml`](src/reviewscope_ml/README.md) | ML pipeline package: segmentation, embedding, clustering, sentiment, labeling, evaluation |
| `src/ollama` | Ollama image that pulls the required models on first start |
| [`notebooks`](notebooks/README.md) | Experiments behind the pipeline decisions |
| [`datasets`](datasets/README.md) | Demo datasets |
| `scripts` | Local development launcher and research run scripts |
| [`docs`](docs/README.md) | Specification, methodology, evaluations |
| `tests` | Tests of the ML package |
