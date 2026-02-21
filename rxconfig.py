import reflex as rx
import os

config = rx.Config(
    app_name="golf_reflex",
    # api_url="http://localhost:8000",
    # deploy_url=os.environ.get("RENDER_EXTERNAL_URL", ""),
    disable_plugins=["reflex.plugins.sitemap.SitemapPlugin"],
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    backend_port=8000,
    frontend_port=3000,
    env=rx.Env.DEV,
)
