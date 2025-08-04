# import sys
# import os

# # Add the parent directory to Python's module search path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from ai_service.question_generator import generate_follow_up_questions
# from ai_service.scorer import calculate_transparency_score
# from backend.schemas import ProductEntry

# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"]
# )

# class ProductEntry(BaseModel):
#     product_name: str
#     category: str
#     ingredients: str = ""
#     origin: str = ""
#     certifications: str = ""

# @app.post("/generate-questions")
# def generate_questions(product: ProductEntry):
#     questions = generate_follow_up_questions(product)
#     return {"follow_up_questions": questions}

# @app.post("/transparency-score")
# def get_transparency_score(product: ProductEntry):
#     result = calculate_transparency_score(product)
#     return result




import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

# ----------------------------
# Schema (ProductEntry)
# ----------------------------

class ProductEntry(BaseModel):
    product_name: str
    category: str
    ingredients: Optional[str] = None
    origin: Optional[str] = None
    certifications: Optional[str] = None

# ----------------------------
# Question Generation Model
# ----------------------------

model_name = "valhalla/t5-base-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

question_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_question_from_context(context: str) -> str:
    input_text = f"generate question: {context}"
    result = question_generator(input_text)
    return result[0]["generated_text"]

def generate_follow_up_questions(product_data: ProductEntry) -> dict:
    prompt = (
        f"Product Name: {product_data.product_name}. "
        f"Category: {product_data.category}. "
        f"Ingredients: {product_data.ingredients or ''}. "
        f"Origin: {product_data.origin or ''}. "
        f"Certifications: {product_data.certifications or ''}. "
        "Given the above product information, list 3 follow-up questions a consumer might ask "
        "to better understand the product. If no additional questions are needed, return an empty list."
    )

    try:
        response = generate_question_from_context(prompt)
        questions = response.strip().split("\n")
        questions = [q.strip("- ").strip() for q in questions if q.strip()]
    except Exception as e:
        print(f"Error parsing questions: {e}")
        questions = [response.strip()]

    return {"follow_up_questions": questions}

# ----------------------------
# Transparency Score Function
# ----------------------------

def calculate_transparency_score(product_data: ProductEntry) -> dict:
    score = 0
    breakdown = {}

    if product_data.ingredients:
        breakdown['ingredient_clarity'] = 90
        score += 30

    if product_data.origin:
        breakdown['origin_disclosure'] = 80
        score += 25

    if product_data.certifications:
        breakdown['certifications'] = 70
        score += 25

    score += 20  # base score
    return {
        "transparency_score": min(score, 100),
        "score_breakdown": breakdown
    }

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/generate-questions")
def generate_questions(product: ProductEntry):
    return generate_follow_up_questions(product)

@app.post("/transparency-score")
def get_transparency_score(product: ProductEntry):
    return calculate_transparency_score(product)
