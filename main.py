import sys
import os

# Add the parent directory to Python's module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_service.question_generator import generate_follow_up_questions
from ai_service.scorer import calculate_transparency_score

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ProductEntry(BaseModel):
    product_name: str
    category: str
    ingredients: str = ""
    origin: str = ""
    certifications: str = ""

@app.post("/generate-questions")
def generate_questions(product: ProductEntry):
    questions = generate_follow_up_questions(product)
    return {"follow_up_questions": questions}

@app.post("/transparency-score")
def get_transparency_score(product: ProductEntry):
    result = calculate_transparency_score(product)
    return result
