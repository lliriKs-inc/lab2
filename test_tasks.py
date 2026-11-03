from __future__ import annotations

import csv
import math
from pathlib import Path
import tempfile
import unittest
import warnings

from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic


ROOT = Path(__file__).resolve().parent
FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def pearson(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return math.nan
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x, _ in pairs)
        * sum((y - mean_y) ** 2 for _, y in pairs)
    )
    return numerator / denominator if denominator else math.nan


class TitanicTests(unittest.TestCase):
    def test_original_data(self):
        result = analyze_titanic(TitanicInput(str(ROOT / "titanic.csv")))
        self.assertEqual(result.row_count, 891)
        self.assertEqual(result.adults_over_30_count, 305)
        self.assertEqual(
            {key: value for key, value in result.missing_by_column.items() if value},
            {"Age": 177, "Cabin": 687, "Embarked": 2},
        )
        self.assertEqual(len(result.missing_by_column), 12)
        for pclass, age, rate in [
            (1, 38.233440860215055, 136 / 216),
            (2, 29.87763005780347, 87 / 184),
            (3, 25.14061971830986, 119 / 491),
        ]:
            self.assertAlmostEqual(result.mean_age_by_pclass[pclass], age)
            self.assertAlmostEqual(result.survival_rate_by_pclass[pclass], rate)
        self.assertTrue(all(type(key) is int for key in result.mean_age_by_pclass))
        self.assertTrue(all(type(key) is int for key in result.survival_rate_by_pclass))
        self.assertEqual(result.highest_fares, [512.3292] * 3 + [263.0] * 2)

    def test_missing_values_threshold_and_duplicate_fares(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passengers.csv"
            path.write_text(
                "Pclass,Age,Survived,Fare\n"
                "1,30,1,50\n1,40,0,50\n1,,1,\n"
                "2,,0,10\n3,31,1,20\n3,0,0,5\n3,29,0,30\n",
                encoding="utf-8",
            )
            result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.row_count, 7)
        self.assertEqual(result.adults_over_30_count, 2)
        self.assertEqual(result.missing_by_column["Age"], 2)
        self.assertEqual(result.missing_by_column["Fare"], 1)
        self.assertEqual(result.mean_age_by_pclass[1], 35.0)
        self.assertTrue(math.isnan(result.mean_age_by_pclass[2]))
        self.assertEqual(result.mean_age_by_pclass[3], 20.0)
        self.assertAlmostEqual(result.survival_rate_by_pclass[1], 2 / 3)
        self.assertEqual(result.highest_fares, [50.0, 50.0, 30.0, 20.0, 10.0])

    def test_fewer_than_five_fares(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            path.write_text("Pclass,Age,Survived,Fare\n2,45,1,7\n", encoding="utf-8")
            result = analyze_titanic(TitanicInput(str(path)))
        self.assertEqual(result.highest_fares, [7.0])
        self.assertEqual(result.mean_age_by_pclass, {2: 45.0})


class BrainCorrelationTests(unittest.TestCase):
    def assert_correlations_match(self, path: Path):
        with path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = analyze_brain_correlations(BrainDataInput(str(path)))
        for gender, actual, count in [
            ("Female", result.women_mri_correlation, result.women_count),
            ("Male", result.men_mri_correlation, result.men_count),
        ]:
            group = [row for row in rows if row["Gender"] == gender]
            self.assertEqual(count, len(group))
            self.assertEqual(set(actual), set(FEATURES))
            for feature in FEATURES:
                pairs = [
                    (float(row[feature]), float(row["MRI_Count"]))
                    for row in group
                    if row[feature] not in ("NA", "")
                    and row["MRI_Count"] not in ("NA", "")
                ]
                expected = pearson(pairs)
                with self.subTest(gender=gender, feature=feature):
                    if math.isnan(expected):
                        self.assertTrue(math.isnan(actual[feature]))
                    else:
                        self.assertAlmostEqual(actual[feature], expected, places=12)
        return result

    def test_original_data_against_pearson_formula(self):
        result = self.assert_correlations_match(ROOT / "brainsize.txt")
        self.assertEqual((result.men_count, result.women_count), (20, 20))
        self.assertEqual(result.strongest_mri_feature, "PIQ")

    def test_pairwise_missing_and_strong_negative_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "brain.tsv"
            path.write_text(
                "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
                "Female\t4\t2\t10\tNA\t1\t1\n"
                "Female\t3\t1\t10\t3\tNA\t2\n"
                "Female\t2\t3\t10\t1\t4\t3\n"
                "Female\t1\t4\t10\t2\t3\t4\n"
                "Male\t1\t4\t1\t1\t4\t3\n"
                "Male\t4\t1\t1\t2\t2\t1\n"
                "Male\t2\t3\t2\t3\t1\t4\n"
                "Male\t3\t2\t3\t4\t3\t2\n"
                "Male\t100\t100\t100\t100\t100\tNA\n",
                encoding="utf-8",
            )
            result = self.assert_correlations_match(path)
        self.assertEqual(result.strongest_mri_feature, "FSIQ")
        self.assertAlmostEqual(result.women_mri_correlation["FSIQ"], -1.0)
        self.assertTrue(math.isnan(result.women_mri_correlation["PIQ"]))

    def test_undefined_correlations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.tsv"
            path.write_text(
                "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
                "Female\t1\t1\t1\t1\t1\t1\n"
                "Male\t1\t1\t1\t1\t1\t1\n",
                encoding="utf-8",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaisesRegex(ValueError, "корреляций"):
                    analyze_brain_correlations(BrainDataInput(str(path)))


if __name__ == "__main__":
    unittest.main()
