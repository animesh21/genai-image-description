import base64

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


app = FastAPI(title="ProductVision")

# CORS for local React dev (adjust origins for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()  # uses OPENAI_API_KEY from env


@app.get('/health')
async def health_check():
    return {'status': 200}


def encode_image_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


@app.post("/api/generate-description")
async def generate_description(
        prompt: str = Form(...),
        model_name: str = Form(...),
        image: UploadFile = File(...),
):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        img_b64 = encode_image_bytes(image_bytes)

        response = client.chat.completions.create(
            model=model_name,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.content_type};base64,{img_b64}"
                            },
                        },
                    ],
                }
            ],
        )

        # In the new client, message content is usually a string
        content = response.choices[0].message.content

        return {
            "description": content,
            "model": response.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log in real app
        raise HTTPException(status_code=500, detail=str(e))
