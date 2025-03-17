import pickle
from enum import Enum

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Fertility Classification API",
    description="API для предсказания диагностики по набору данных Fertility",
    version="1.0.0"
)

# Загрузка модели
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Загрузка списка ожидаемых столбцов
with open("expected_columns.pkl", "rb") as file:
    expected_columns = pickle.load(file)


# Enums для справочников
class SeasonEnum(str, Enum):
    WINTER = "Зима"
    SPRING = "Весна"
    SUMMER = "Лето"
    FALL = "Осень"


class HighFeversEnum(str, Enum):
    LESS_THAN_3_MONTHS = "Менее 3 месяцев назад"
    MORE_THAN_3_MONTHS = "Более 3 месяцев назад"
    NO_FEVER = "Нет"


class SmokingEnum(str, Enum):
    NEVER = "Никогда"
    OCCASIONAL = "Иногда"
    DAILY = "Ежедневно"


class AlcoholEnum(str, Enum):
    SEVERAL_TIMES_A_DAY = "Несколько раз в день"
    EVERY_DAY = "Каждый день"
    SEVERAL_TIMES_A_WEEK = "Несколько раз в неделю"
    ONCE_A_WEEK = "Раз в неделю"
    HARDLY_EVER = "Редко или никогда"


class DiagnosisEnum(str, Enum):
    NORMAL = "Normal"
    OLIGOSPERMIA = "Oligospermia"


# Модель, где храним все поля (как было в документации)
class Features(BaseModel):
    age: float = Field(..., ge=0, le=1, example=0.5, description="Возраст (нормализованный от 0 до 1)")
    child_diseases: int = Field(..., ge=0, le=1, example=1, description="Болезни в детстве (1 - Да, 0 - Нет)")
    accident: int = Field(..., ge=0, le=1, example=0, description="Аварии или серьёзные травмы (1 - Да, 0 - Нет)")
    surgical_intervention: int = Field(..., ge=0, le=1, example=0,
                                       description="Хирургические вмешательства (1 - Да, 0 - Нет)")
    high_fevers: HighFeversEnum = Field(..., example="Нет", description="Высокая температура за последний год")
    alcohol: AlcoholEnum = Field(..., example="Редко или никогда", description="Частота потребления алкоголя")
    smoking: SmokingEnum = Field(..., example="Никогда", description="Курение")
    hrs_sitting: float = Field(..., ge=0, le=1, example=0.4,
                               description="Часы сидячей работы (нормализованные от 0 до 1)")
    season: SeasonEnum = Field(..., example="Лето", description="Сезон анализа")


# Теперь FertilityInput содержит одно поле features
class FertilityInput(BaseModel):
    features: Features


# Маппинги
season_mapping = {
    "Зима": -1,
    "Весна": -0.33,
    "Лето": 0.33,
    "Осень": 1
}
high_fevers_mapping = {
    "Менее 3 месяцев назад": -1,
    "Более 3 месяцев назад": 0,
    "Нет": 1
}
smoking_mapping = {
    "Никогда": -1,
    "Иногда": 0,
    "Ежедневно": 1
}
alcohol_mapping = {
    "Несколько раз в день": 0,
    "Каждый день": 0.2,
    "Несколько раз в неделю": 0.4,
    "Раз в неделю": 0.6,
    "Редко или никогда": 1
}


@app.post("/predict")
def predict(input_data: FertilityInput):
    # Получаем словарь features
    data_dict = input_data.features.dict()

    # Преобразуем значения
    data_dict["season"] = season_mapping[data_dict["season"]]
    data_dict["high_fevers"] = high_fevers_mapping[data_dict["high_fevers"]]
    data_dict["smoking"] = smoking_mapping[data_dict["smoking"]]
    data_dict["alcohol"] = alcohol_mapping[data_dict["alcohol"]]

    # Создаём DataFrame
    input_df = pd.DataFrame([data_dict])

    # One-Hot Encoding
    input_df_encoded = pd.get_dummies(input_df, drop_first=True)
    input_df_encoded = input_df_encoded.reindex(columns=expected_columns, fill_value=0)

    # Предсказание
    proba = model.predict_proba(input_df_encoded)[0]
    prob_positive = proba[1]  # вероятность "Oligospermia"
    prob_negative = proba[0]  # вероятность "Normal"

    threshold = 0.5
    if prob_positive >= threshold:
        diagnosis = DiagnosisEnum.OLIGOSPERMIA.value
        diagnosis_probability = prob_positive
    else:
        diagnosis = DiagnosisEnum.NORMAL.value
        diagnosis_probability = prob_negative

    diagnosis_probability = round(float(diagnosis_probability), 4)

    return {
        "diagnosis": diagnosis,
        "probability": diagnosis_probability
    }
