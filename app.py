import os

from flask import Flask, jsonify, request, send_from_directory

from engine import Engine

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024      # 12 MB uploads

engine = Engine()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    return jsonify(status="ok", model_ready=engine.ready)


@app.get("/api/info")
def info():
    return jsonify(ready=engine.ready, error=engine.error, model=engine.info)


@app.post("/api/check")
def check():
    if not engine.ready:
        return jsonify(error=engine.error or "The model isn’t loaded on this server."), 503
    f = request.files.get("image")
    if f is None or not f.filename:
        return jsonify(error="No photo received. Choose an image file."), 400
    try:
        return jsonify(engine.check(f.read()))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception:
        app.logger.exception("check failed")
        return jsonify(error="The check failed on the server. Try again, or use a smaller photo."), 500


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="That photo is over 12 MB. Choose a smaller one."), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
