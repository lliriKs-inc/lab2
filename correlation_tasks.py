"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    observations = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    men = observations.loc[observations["Gender"] == "Male"]
    women = observations.loc[observations["Gender"] == "Female"]

    def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
        correlations = group[features].corrwith(group["MRI_Count"], method="pearson")
        return {feature: float(value) for feature, value in correlations.items()}

    men_correlations = mri_correlations(men)
    women_correlations = mri_correlations(women)
    valid_correlations = [
        (feature, value)
        for correlations in (women_correlations, men_correlations)
        for feature, value in correlations.items()
        if pd.notna(value)
    ]
    if not valid_correlations:
        raise ValueError("Нет определённых корреляций признаков с MRI_Count.")

    strongest_feature, _ = max(valid_correlations, key=lambda item: abs(item[1]))
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_correlations,
        men_mri_correlation=men_correlations,
        strongest_mri_feature=strongest_feature,
    )
