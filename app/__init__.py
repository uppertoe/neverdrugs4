from flask import Flask, jsonify


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> tuple[dict[str, str], int]:
        # Basic readiness probe for infrastructure tests
        return jsonify(status="ok"), 200

    return app
