"""User interface package for EdgeMate."""

from .gradio_app import create_app
from .web_app import create_web_app

__all__ = ["create_app", "create_web_app"]
