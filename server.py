import os
import asyncio
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class ImageRequest(BaseModel):
    prompts: list[str]
    provider: str  # "openai" or "gemini"


class ImageResult(BaseModel):
    prompt: str
    image_base64: str | None = None
    error: str | None = None


class ImageResponse(BaseModel):
    results: list[ImageResult]


async def generate_openai(prompt: str) -> ImageResult:
    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = await client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )
        return ImageResult(prompt=prompt, image_base64=response.data[0].b64_json)
    except Exception as e:
        return ImageResult(prompt=prompt, error=str(e))


async def generate_gemini(prompt: str) -> ImageResult:
    try:
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = await asyncio.to_thread(
            client.models.generate_images,
            model="imagen-3.0-generate-002",
            prompt=prompt,
            config=genai.types.GenerateImagesConfig(number_of_images=1),
        )
        if response.generated_images:
            img_bytes = response.generated_images[0].image.image_bytes
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            return ImageResult(prompt=prompt, image_base64=b64)
        else:
            return ImageResult(prompt=prompt, error="No image generated")
    except Exception as e:
        return ImageResult(prompt=prompt, error=str(e))


@app.post("/api/generate", response_model=ImageResponse)
async def generate_images(request: ImageRequest):
    prompts = [p for p in request.prompts if p.strip()]
    if not prompts:
        raise HTTPException(status_code=400, detail="No prompts provided")

    if request.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
        gen_fn = generate_openai
    elif request.provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise HTTPException(status_code=400, detail="GEMINI_API_KEY not set")
        gen_fn = generate_gemini
    else:
        raise HTTPException(status_code=400, detail="Invalid provider")

    tasks = [gen_fn(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)

    return ImageResponse(results=list(results))


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
