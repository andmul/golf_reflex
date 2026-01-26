import reflex as rx
import os

config = rx.Config(
    app_name="golf_reflex",  # Must match your app class name
    api_url=os.environ.get("RENDER_EXTERNAL_URL", ""),  # Critical!
    deploy_url=os.environ.get("RENDER_EXTERNAL_URL", ""),
    cors_allowed_origins=[
        os.environ.get("RENDER_EXTERNAL_URL", ""),
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    backend_port=8000,
    frontend_port=3000,
    # Enable these for production
    env=rx.Env.PROD,
)
