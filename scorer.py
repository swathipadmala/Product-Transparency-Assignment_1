def calculate_transparency_score(product_data: dict) -> dict:
    score = 0
    breakdown = {}

    if 'ingredients' in product_data:
        breakdown['ingredient_clarity'] = 90
        score += 30

    if 'origin' in product_data:
        breakdown['origin_disclosure'] = 80
        score += 25

    if 'certifications' in product_data:
        breakdown['certifications'] = 70
        score += 25

    score += 20  # base score
    return {
        "transparency_score": min(score, 100),
        "score_breakdown": breakdown
    }
