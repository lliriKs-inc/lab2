"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    passengers = pd.read_csv(data.csv_path)
    by_class = passengers.groupby("Pclass")

    return TitanicSummary(
        row_count=len(passengers),
        missing_by_column={
            column: int(count) for column, count in passengers.isna().sum().items()
        },
        adults_over_30_count=int((passengers["Age"] > 30).sum()),
        mean_age_by_pclass={
            int(pclass): float(age)
            for pclass, age in by_class["Age"].mean().items()
        },
        survival_rate_by_pclass={
            int(pclass): float(rate)
            for pclass, rate in by_class["Survived"].mean().items()
        },
        highest_fares=[
            float(fare) for fare in passengers["Fare"].dropna().nlargest(5)
        ],
    )
