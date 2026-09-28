from pathlib import Path

path = Path(r".\backend\app.py")
text = path.read_text(encoding="utf-8")

marker = '''ALLOWED_PREDICTIONS = {
'''

recommendation_block = '''RECOMMENDATIONS = {
    "Bacterial_spot": (
        "Remove badly affected leaves, avoid overhead watering, "
        "and keep foliage dry with good airflow."
    ),
    "Early_blight": (
        "Remove affected leaves, improve airflow, avoid overhead watering, "
        "and monitor nearby leaves for spreading symptoms."
    ),
    "Late_blight": (
        "Remove affected plant material, improve airflow, avoid prolonged "
        "leaf wetness, and isolate heavily affected plants when practical."
    ),
    "Leaf_Mold": (
        "Improve ventilation and reduce leaf humidity, especially around "
        "dense foliage. Remove severely affected leaves."
    ),
    "powdery_mildew": (
        "Improve airflow and sunlight exposure, remove severely affected "
        "leaves, and avoid excessive humidity around the foliage."
    ),
    "Septoria_leaf_spot": (
        "Remove affected leaves, avoid splashing water onto foliage, "
        "and improve airflow around the plant."
    ),
    "Spider_mites Two-spotted_spider_mite": (
        "Inspect the undersides of leaves, wash foliage where appropriate, "
        "and monitor mite levels and plant stress."
    ),
    "Target_Spot": (
        "Remove severely affected leaves, reduce leaf wetness, "
        "and improve airflow around the plant."
    ),
    "Tomato_mosaic_virus": (
        "Remove and isolate visibly infected plants where practical, "
        "sanitize tools, and avoid handling plants when foliage is wet."
    ),
    "Tomato_Yellow_Leaf_Curl_Virus": (
        "Inspect for whitefly activity, remove severely affected plants "
        "where practical, and manage the insect vector."
    ),
    "healthy": (
        "No disease was detected with sufficient confidence. "
        "Continue regular monitoring and good plant-care practices."
    ),
}


'''

if marker not in text:
    raise SystemExit(
        "ALLOWED_PREDICTIONS marker was not found. No changes made."
    )

if "RECOMMENDATIONS = {" in text:
    raise SystemExit(
        "RECOMMENDATIONS already exists. No changes made."
    )

text = text.replace(
    marker,
    recommendation_block + marker,
    1,
)

path.write_text(text, encoding="utf-8")

print("Recommendation mapping added.")
print(f"Recommendations configured: {len(recommendation_block.split('\"'))}")
