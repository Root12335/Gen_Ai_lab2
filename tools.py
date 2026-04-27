import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

from schemas import MealRecord


load_dotenv()
CSV_PATH = Path(__file__).resolve().parent / "meals_log.csv"
CSV_HEADERS = [
    "date",
    "meal_text",
    "image_note",
    "estimated_calories",
    "protein_g",
    "carbs_g",
    "fat_g",
    "summary",
    "goal",
    "city",
]


def ensure_csv_exists() -> None:
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)


def _make_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment variables.")
    return TavilyClient(api_key=api_key)


@tool
def search_healthy_options(query: str, city: str = "") -> str:
    """Search healthy restaurants, grocery stores, and nutrition info."""
    client = _make_tavily_client()
    search_query = f"{query} in {city}".strip()
    result = client.search(search_query, max_results=5)
    return str(result)


@tool
def store_meal_record(
    meal_text: str,
    image_note: str = "",
    estimated_calories: float = 0.0,
    protein_g: float = 0.0,
    carbs_g: float = 0.0,
    fat_g: float = 0.0,
    summary: str = "",
    goal: str = "",
    city: str = "",
) -> str:
    """Store meal analysis in CSV with calories, macros, summary, and goals."""
    ensure_csv_exists()
    record = MealRecord(
        meal_text=meal_text,
        image_note=image_note,
        estimated_calories=estimated_calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        summary=summary,
        goal=goal,
        city=city,
    )
    with CSV_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                record.date,
                record.meal_text,
                record.image_note,
                record.estimated_calories,
                record.protein_g,
                record.carbs_g,
                record.fat_g,
                record.summary,
                record.goal,
                record.city,
            ]
        )
    return f"Meal record stored successfully in {CSV_PATH.name}."

