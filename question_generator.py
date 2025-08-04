from .model import generate_question_from_context
from backend.schemas import ProductEntry  # Adjust the import based on your actual structure

def generate_follow_up_questions(product_data: ProductEntry) -> dict:
    prompt = (
        f"Product Name: {product_data.product_name}. "
        f"Category: {product_data.category}. "
        f"Ingredients: {product_data.ingredients}. "
        f"Origin: {product_data.origin}. "
        f"Certifications: {product_data.certifications}. "
        "Given the above product information, list 3 follow-up questions a consumer might ask "
        "to better understand the product. If no additional questions are needed, return an empty list."
    )

    # Get follow-up question suggestions from model
    response = generate_question_from_context(prompt)

    # Clean and split questions
    try:
        questions = response.strip().split("\n")
        questions = [q.strip("- ").strip() for q in questions if q.strip()]
    except Exception as e:
        print(f"Error parsing questions: {e}")
        questions = [response.strip()]

    return {"follow_up_questions": questions}
