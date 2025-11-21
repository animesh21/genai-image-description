import base64

import uvicorn
from openai import OpenAI


def encode_image(path: str) -> str:
    """Reads image and returns base64 encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    client = OpenAI()

    # Path to the image you want to send
    image_path = "images/image_1.jpg"
    img_b64 = encode_image(image_path)

    # Create the request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": ("1. Please analyze the image carefully and find a product in it(there can be multiple "
                             "smaller objects which represent a product as well).\n"
                             "2. Now, once identified the product, can you please generate concise, accurate, "
                             "and customer friendly product description for the product ?\n"
                             "3. Please don't hallucinate, if no product is idenfiable, just respond with `NA`.")
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                }
            ]}
        ]
    )

    print("\n=== Model Response ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
