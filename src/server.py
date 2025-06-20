from flask_failsafe import failsafe

@failsafe
def create_app():
    from app import app
    return app.server

if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8050)
