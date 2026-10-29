# Fishy: refund photo check

Upload a photo, get a tampering score and a heatmap of the most suspicious regions.
Flask + PyTorch (CPU). Same pipeline as the notebook: uniform 512px JPEG, frozen ResNet50
(layer3 + layer4 on an 8x8 grid), patch scorer, mean of the top 4 patch scores.

## 1. Export the trained model from the notebook (once)
Paste `export_cell.py` as the last cell of the Colab notebook and run it (after section 13).
It downloads `fishy_model.pt` (under 1 MB). Put it in `model/`, so you have `model/fishy_model.pt`.
Check that it prints `max logit difference` close to 0.

## 2. Try it locally (optional)
    pip install -r requirements.txt
    python fetch_backbone.py
    python app.py            # http://localhost:5000

## 3. Deploy on Render
1. Push this folder to a GitHub repo (include `model/fishy_model.pt`).
2. Render > New > Blueprint > pick the repo (it reads `render.yaml`).
   Or New > Web Service with:
   - Build: `pip install -r requirements.txt && python fetch_backbone.py`
   - Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180`
   - Env var: `PYTHON_VERSION=3.11.9`
3. Open the URL. `/health` should say `model_ready: true`.

Free-plan notes: 512 MB RAM is tight (if the log says "out of memory", use a bigger instance or
Hugging Face Spaces below). The service sleeps when idle, so the first request takes about a minute.
CPU is slow there, so one check can take 10 to 30 seconds.

## Alternative: Hugging Face Spaces (more free RAM)
New Space > SDK: Docker > upload these files. Add this to the top of README.md:

    ---
    title: Fishy
    sdk: docker
    app_port: 7860
    ---

## Limits (say these in the demo)
This is a flag-for-human-review tool. "Not flagged" does not mean genuine. Cross-validated on
100 real photos plus fakes from Stable Diffusion, Gemini and ChatGPT. See the in-app "About this model".
