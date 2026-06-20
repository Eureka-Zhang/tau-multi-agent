# Copyright Sierra

import os

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file (API keys, base URLs) as early as
# possible, so every entry point and every litellm call sees them. find_dotenv
# walks up from the current working directory to locate the nearest .env.
load_dotenv(find_dotenv(usecwd=True))

# litellm / the OpenAI SDK read the base URL from different env vars depending on
# version. Mirror whichever one is provided so an OpenAI-compatible endpoint
# (custom proxy/gateway) works regardless of naming convention.
_openai_base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
    "OPENAI_API_BASE"
)
if _openai_base_url:
    os.environ.setdefault("OPENAI_BASE_URL", _openai_base_url)
    os.environ.setdefault("OPENAI_API_BASE", _openai_base_url)

from tau_bench.envs.base import Env as Env
from tau_bench.agents.base import Agent as Agent
