from datetime import datetime
from pydantic import BaseModel, Field


class MealRecord(BaseModel):
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    meal_text: str
    image_note: str = ""
    estimated_calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    summary: str = ""
    goal: str = ""
    city: str = ""

