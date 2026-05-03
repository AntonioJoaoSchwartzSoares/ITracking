# racing-line-ai

Extração de traçado (racing line) a partir de vídeo aéreo estático: YOLO + tracking, trajetória em CSV, overlay em vídeo e mapa de calor.

## Requisitos

- Python 3.10+ **ou** [Docker Desktop](https://www.docker.com/products/docker-desktop/) com GPU (opcional, recomendado).
- [Git](https://git-scm.com/download/win) para clonar e enviar alterações.

## Uso rápido (Python local)

```powershell
cd racing-line-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\extract_line.py --video "data\input\seu_video.mp4" --out "data\saida" --class-id -1 --auto-select --start-x 640 --start-y 400
```

Scripts principais:

| Ficheiro | Descrição |
|----------|-----------|
| `extract_line.py` | Versão simples na raiz (uma deteção forte por frame + fallback CV). |
| `src/extract_line.py` | Tracking com IDs, `--auto-select`, `--exclude-rect`, `--target-id`, etc. |
| `src/extract_line_inicial.py` | Variante próxima do exemplo didático (PNG + CSV). |

## Docker

```powershell
cd racing-line-ai
.\scripts\docker-run.cmd -Video ".\data\input\seu_video.mp4" -Out ".\data\saida" --class-id -1
```

Com o script completo: `.\scripts\docker-run.cmd` sem `-UseOriginal` (usa `src/extract_line.py`). Ver comentários no topo de `scripts/docker-run.ps1`.

## Dados

Coloca vídeos em `data/input/`. Pastas em `data/output*` são ignoradas pelo Git (geradas localmente). Datasets YOLO em `datasets/` também ficam de fora do repositório por tamanho; podes voltar a exportar do Roboflow ou anexar com [Git LFS](https://git-lfs.com/) se precisares no remoto.

## Licença

Definir conforme a tua preferência (ex.: MIT).
