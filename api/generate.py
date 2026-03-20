import os
import asyncio
import base64
import json
from http.server import BaseHTTPRequestHandler


def generate_openai_sync(prompt: str) -> dict:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        return {"prompt": prompt, "image_base64": response.data[0].b64_json}
    except Exception as e:
        return {"prompt": prompt, "error": str(e)}


def generate_gemini_sync(prompt: str) -> dict:
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_images(
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(number_of_images=1),
        )
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            return {"prompt": prompt, "image_base64": b64}
        else:
            return {"prompt": prompt, "error": "No image generated"}
    except Exception as e:
        return {"prompt": prompt, "error": str(e)}


async def generate_openai(prompt: str) -> dict:
    return await asyncio.to_thread(generate_openai_sync, prompt)


async def generate_gemini(prompt: str) -> dict:
    return await asyncio.to_thread(generate_gemini_sync, prompt)


async def handle_generate(prompts: list[str], provider: str) -> list[dict]:
    if provider == "openai":
        gen_fn = generate_openai
    else:
        gen_fn = generate_gemini

    tasks = [gen_fn(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        prompts = [p for p in data.get("prompts", []) if p.strip()]
        provider = data.get("provider", "openai")

        if not prompts:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "No prompts provided"}).encode())
            return

        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "OPENAI_API_KEY not set"}).encode())
            return

        if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "GEMINI_API_KEY not set"}).encode())
            return

        results = asyncio.run(handle_generate(prompts, provider))

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"results": results}).encode())
