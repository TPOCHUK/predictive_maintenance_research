from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "notebooks_incoming"
TARGET = ROOT / "notebooks"
ARCHIVE = TARGET / "archive"


ORDER = [
    ("main_dataset.ipynb", "01_dataset_preparation.ipynb", "Підготовка даних для моделей", "explore"),
    ("main.ipynb", "02_feature_search_xgboost.ipynb", "Пошук ознак і базова модель XGBoost", "explore"),
    ("Wavelets.ipynb", "03_wavelet_features.ipynb", "Wavelet-ознаки", "explore"),
    ("custom.ipynb", "04_custom_temporal_operator.ipynb", "Власна ознака: Custom Temporal Operator", "explore"),
    ("Degradation_Detector_v1.ipynb", "05_degradation_detector.ipynb", "Детектор деградації", "explore"),
    ("Compose.ipynb", "06_composed_features.ipynb", "Комбінація інженерних ознак", "explore"),
    ("LSTM.ipynb", "07_lstm_experiments.ipynb", "Експерименти з LSTM", "middle"),
    ("1D-CNN.ipynb", "08_cnn_experiments.ipynb", "Експерименти з 1D-CNN", "middle"),
    ("main_test_w80.ipynb", "09_official_test_w80.ipynb", "Порівняння на official test для w80", "final"),
    ("train_partial_history_colab.ipynb", "10_partial_history_training.ipynb", "Навчання на часткових історіях", "final"),
    ("train_conservative_lstm_q20_colab.ipynb", "11_conservative_lstm_q20.ipynb", "Консервативна LSTM з q20", "final"),
    ("train_conservative_lstm_q20_calibrated_colab.ipynb", "12_conservative_lstm_q20_calibrated.ipynb", "Калібрована консервативна LSTM", "final"),
]


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text.strip() + "\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text.strip() + "\n"),
    }


def ensure_cell_ids(notebook: dict) -> dict:
    """Assign stable nbformat cell IDs so future nbformat versions stay warning-free."""
    seen: set[str] = set()
    for index, cell in enumerate(notebook.get("cells", [])):
        digest = hashlib.sha1(
            f"{index}|{cell.get('cell_type')}|{''.join(cell.get('source', []))}".encode("utf-8")
        ).hexdigest()[:12]
        while digest in seen:
            digest = hashlib.sha1((digest + "x").encode("ascii")).hexdigest()[:12]
        cell["id"] = digest
        seen.add(digest)
    return notebook


def intro(index: int, title: str, phase: str) -> str:
    nav = f"**Етап {index:02d} з 13** · NASA C-MAPSS FD001 · RUL prediction"
    if phase == "explore":
        return f"""# {index:02d}. {title}

{nav}

> 🧪 **Лабораторний журнал.** На цьому етапі я свідомо залишаю кілька паралельних гіпотез. Тут ще є трохи дослідницького хаосу: швидкі перевірки, невдалі напрямки й нотатки про те, що варто спробувати далі.

**Чернетка думок:** що дасть сигнал, що може виявитися шумом, і чи не з'являється leakage. Рішення фіксую після таблиці результатів, а не до експерименту.
"""
    if phase == "middle":
        return f"""# {index:02d}. {title}

{nav}

На цьому етапі простір гіпотез уже звужений. Я порівнюю архітектури та вікна за однаковим engine-level split, фіксую seed і не переношу висновки з validation на official test без окремої перевірки.
"""
    return f"""# {index:02d}. {title}

{nav}

**Мета.** Відтворювано оцінити модель за зафіксованим протоколом і зберегти артефакти для подальшого використання.

**Правило оцінювання.** Official test використовується один раз. Raw і RUL-capped метрики подаються окремо та не змішуються в одному рейтингу.
"""


def clean_outputs(cell: dict) -> None:
    if cell.get("cell_type") != "code":
        return
    source = "".join(cell.get("source", [])).lower()
    outputs = cell.get("outputs", [])
    text = " ".join(
        "".join(output.get("text", []))
        + "".join(output.get("data", {}).get("text/plain", []))
        for output in outputs
    )
    transient = (
        source.lstrip().startswith("!pip")
        or ("mlflow.set_tracking_uri" in source and "fit(" not in source)
        or "traceback" in text.lower()
        or (len(text) > 45000 and "dataframe" not in text.lower())
    )
    if transient:
        cell["outputs"] = []
        cell["execution_count"] = None


MLFLOW_HELPER = r'''
# Optional MLflow model registry integration.
# In Colab set MLFLOW_TRACKING_URI in the environment (or leave it empty to skip logging).
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT", "Predictive Maintenance - Conservative LSTM"
)


def _mlflow_metric_name(name):
    return re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")


def log_torch_model_to_mlflow(model, run_name, params, metrics, model_name="model"):
    """Log a complete PyTorch model plus reproducibility metadata; no-op when URI is empty."""
    if not MLFLOW_TRACKING_URI:
        print("MLflow skipped: set MLFLOW_TRACKING_URI to enable remote logging.")
        return None

    try:
        import mlflow
        import mlflow.pytorch
    except ImportError as exc:
        raise RuntimeError("Install MLflow first: %pip install -q mlflow") from exc

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    serializable_metrics = {
        _mlflow_metric_name(key): float(value)
        for key, value in metrics.items()
        if np.isscalar(value) and np.isfinite(value)
    }
    model_for_log = copy.deepcopy(model).cpu().eval()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(serializable_metrics)
        mlflow.log_dict(
            {"features": list(FEATURE_COLUMNS)}, "reproducibility/feature_columns.json"
        )
        try:
            mlflow.pytorch.log_model(model_for_log, name=model_name)
        except TypeError:  # MLflow < 3
            mlflow.pytorch.log_model(model_for_log, artifact_path=model_name)
        print("MLflow run:", run.info.run_id)
        return run.info.run_id
'''


Q20_LOG = r'''
mlflow_run_id = log_torch_model_to_mlflow(
    model=final_model,
    run_name="FINAL | conservative_lstm_q20_w80",
    params={
        "dataset": "NASA C-MAPSS FD001",
        "architecture": "RULLSTM",
        "quantile": QUANTILE,
        "window_size": WINDOW_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "selected_epochs": best_epoch,
        "partial_histories": True,
        "seed": SEED,
    },
    metrics={f"raw_{key}": value for key, value in raw_metrics.items()},
    model_name="conservative_lstm_q20",
)
'''


CALIBRATED_LOG = r'''
common_params = {
    "dataset": "NASA C-MAPSS FD001",
    "architecture": "RULLSTM",
    "quantile": QUANTILE,
    "window_size": WINDOW_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "selected_epochs": best_epoch,
    "train_engines": 64,
    "validation_engines": 16,
    "calibration_engines": 20,
    "seed": SEED,
}

base_mlflow_run_id = log_torch_model_to_mlflow(
    model=final_model,
    run_name="FINAL | conservative_lstm_q20_retrained_base",
    params={**common_params, "calibrated": False},
    metrics={f"raw_{key}": value for key, value in rows["Base P20 raw"].items()},
    model_name="base_conservative_lstm_q20",
)

calibrated_mlflow_run_id = log_torch_model_to_mlflow(
    model=calibrated_model,
    run_name="FINAL | conservative_lstm_q20_calibrated",
    params={
        **common_params,
        "calibrated": True,
        "target_coverage": TARGET_COVERAGE,
        "safety_margin": safety_margin,
    },
    metrics={f"raw_{key}": value for key, value in rows["Calibrated P20 raw"].items()},
    model_name="calibrated_conservative_lstm_q20",
)
'''


def insert_mlflow(notebook: dict, calibrated: bool) -> None:
    cells = notebook["cells"]
    import_index = next(
        i for i, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and "import torch" in "".join(cell.get("source", []))
    )
    # The helper uses re for safe MLflow metric names.
    import_source = "".join(cells[import_index]["source"])
    if not re.search(r"(?m)^import re\s*$", import_source):
        updated = import_source.replace("import random", "import random\nimport re")
        import_source = updated if updated != import_source else "import re\n" + import_source
        cells[import_index]["source"] = lines(import_source)
    cells.insert(import_index + 1, markdown("### MLflow: логування саме моделей\n\nОбгортка нижче є опційною: без `MLFLOW_TRACKING_URI` ноутбук працює локально й не падає. Якщо URI заданий, у run потрапляють параметри, метрики, список ознак і повна PyTorch-модель."))
    cells.insert(import_index + 2, code(MLFLOW_HELPER))

    evaluation_index = next(
        i for i, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "report = pd.DataFrame" in "".join(cell.get("source", []))
    )
    cells.insert(evaluation_index + 1, markdown("### Реєстрація фінальних моделей у MLflow\n\nЦей крок виконується після одноразової оцінки, щоб у MLflow збереглися саме фінальні ваги, а не проміжна модель із model-selection."))
    cells.insert(evaluation_index + 2, code(CALIBRATED_LOG if calibrated else Q20_LOG))


def curate(source_name: str, target_name: str, title: str, phase: str, index: int) -> None:
    notebook = json.loads((SOURCE / source_name).read_text(encoding="utf-8"))
    cells = [cell for cell in notebook.get("cells", []) if not (
        cell.get("cell_type") == "code" and not "".join(cell.get("source", [])).strip()
    )]
    for cell in cells:
        clean_outputs(cell)
    if cells and cells[0].get("cell_type") == "markdown":
        cells[0] = markdown(intro(index, title, phase))
    else:
        cells.insert(0, markdown(intro(index, title, phase)))
    notebook["cells"] = cells
    notebook.setdefault("metadata", {})["project_order"] = index
    notebook["metadata"]["project_phase"] = phase
    notebook["metadata"]["language"] = "uk"
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    if source_name == "train_conservative_lstm_q20_colab.ipynb":
        insert_mlflow(notebook, calibrated=False)
    if source_name == "train_conservative_lstm_q20_calibrated_colab.ipynb":
        insert_mlflow(notebook, calibrated=True)
    ensure_cell_ids(notebook)
    (TARGET / target_name).write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def final_comparison_notebook() -> dict:
    cells = [
        markdown("""# 13. Фінальне порівняння моделей

**Етап 13 з 13** · NASA C-MAPSS FD001 · підсумок дослідження

Цей ноутбук не перенавчає моделі. Він збирає зафіксовані результати з попередніх експериментів, відокремлює raw від capped протоколу та формулює фінальний вибір.
"""),
        markdown("""## 1. Як читати порівняння

- **Raw official test** — головна таблиця для чесного зіставлення моделей.
- **Capped at 125** — додаткова прикладна оцінка; її не можна напряму ранжувати разом з raw.
- Результати validation і official test не змішуються.
- Калібрована q20 навчалася за іншим split (64/16/20), тому її коректно порівнювати насамперед із retrained base з того самого запуску.
"""),
        code("""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8-whitegrid")
"""),
        markdown("## 2. Raw official-test результати"),
        code("""raw_results = pd.DataFrame([
    {"model": "LSTM w80", "protocol": "full-window", "MAE": 10.430, "RMSE": 16.288, "R2": 0.846, "NASA_Score": np.nan, "over_rate": np.nan, "source": "09_official_test_w80"},
    {"model": "XGBoost", "protocol": "last-cycle tabular", "MAE": 15.571, "RMSE": 22.180, "R2": 0.715, "NASA_Score": np.nan, "over_rate": np.nan, "source": "09_official_test_w80"},
    {"model": "1D-CNN w80", "protocol": "full-window", "MAE": 15.884, "RMSE": 28.043, "R2": 0.545, "NASA_Score": np.nan, "over_rate": np.nan, "source": "09_official_test_w80"},
    {"model": "LSTM partial-history", "protocol": "prefix training", "MAE": 13.118, "RMSE": 20.376, "R2": 0.760, "NASA_Score": np.nan, "over_rate": np.nan, "source": "10_partial_history_training"},
    {"model": "CNN partial-history", "protocol": "prefix training", "MAE": 17.661, "RMSE": 25.757, "R2": 0.616, "NASA_Score": np.nan, "over_rate": np.nan, "source": "10_partial_history_training"},
    {"model": "Conservative LSTM q20", "protocol": "prefix + quantile", "MAE": 9.913, "RMSE": 15.060, "R2": 0.869, "NASA_Score": 466.107, "over_rate": 0.460, "source": "11_conservative_lstm_q20"},
]).sort_values("MAE").reset_index(drop=True)

display(raw_results.round(3))
"""),
        code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ordered = raw_results.sort_values("MAE", ascending=True)
axes[0].barh(ordered["model"], ordered["MAE"], color="#315c8a")
axes[0].set_xlabel("MAE (cycles, lower is better)")
axes[0].set_title("Raw official test")
axes[1].scatter(raw_results["MAE"], raw_results["RMSE"], s=80, color="#b35c2e")
for _, row in raw_results.iterrows():
    axes[1].annotate(row["model"], (row["MAE"], row["RMSE"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
axes[1].set_xlabel("MAE")
axes[1].set_ylabel("RMSE")
axes[1].set_title("Середня й велика помилка")
plt.tight_layout()
plt.show()
"""),
        markdown("## 3. q20: ціна додаткової обережності"),
        code("""calibration_results = pd.DataFrame([
    {"model": "Retrained base q20", "MAE": 11.677, "RMSE": 17.456, "R2": 0.824, "NASA_Score": 783.084, "over_rate": 0.30, "coverage": 0.70, "mean_bias": -2.031},
    {"model": "Calibrated q20", "MAE": 12.801, "RMSE": 17.881, "R2": 0.815, "NASA_Score": 703.643, "over_rate": 0.24, "coverage": 0.76, "mean_bias": -4.372},
])
display(calibration_results.round(3))

delta = calibration_results.set_index("model").loc["Calibrated q20"] - calibration_results.set_index("model").loc["Retrained base q20"]
display(delta.to_frame("calibrated_minus_base").round(3))
"""),
        markdown("""## 4. Фінальний вибір

**Найкраща точність:** `Conservative LSTM q20` з окремого q20 запуску — MAE 9.913, RMSE 15.060 і NASA Score 466.107 на raw official test.

**Найкращий контроль завищення в каліброваній парі:** `Calibrated q20` — overestimation rate зменшився з 0.30 до 0.24, coverage зріс з 0.70 до 0.76, NASA Score знизився з 783.084 до 703.643. Ціна цього рішення — MAE зріс приблизно на 1.12 циклу.

Для демонстрації системи я б залишив обидва артефакти: q20 як основну модель точності та calibrated q20 як safety-режим. Остаточний вибір залежить від вартості хибного завищення RUL у конкретній установці.
"""),
        markdown("""## 5. Обмеження

FD001 має один режим роботи й один тип деградації. Результати не доводять, що модель так само працюватиме на FD002–FD004 або на реальному обладнанні. Наступний крок — повторити протокол на кількох піднаборах C-MAPSS, додати bootstrap-інтервали та перевірити стабільність q20 за кількома seed.
"""),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "project_order": 13,
            "project_phase": "final",
            "language": "uk",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return ensure_cell_ids(notebook)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for index, (source, target, title, phase) in enumerate(ORDER, start=1):
        curate(source, target, title, phase, index)

    legacy = json.loads((SOURCE / "main_test_old_draft.ipynb").read_text(encoding="utf-8"))
    legacy["cells"].insert(0, markdown("# Архівна чернетка\n\nЦей ноутбук збережено лише для історії. Його замінюють етапи 09–13; не використовуйте його як джерело фінальних метрик."))
    legacy["nbformat"] = 4
    legacy["nbformat_minor"] = 5
    ensure_cell_ids(legacy)
    (ARCHIVE / "00_legacy_official_test_draft.ipynb").write_text(
        json.dumps(legacy, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    final_path = TARGET / "13_final_model_comparison.ipynb"
    final_path.write_text(
        json.dumps(final_comparison_notebook(), ensure_ascii=False, indent=1), encoding="utf-8"
    )

    keep = {target for _, target, _, _ in ORDER} | {final_path.name, "README.md", "archive"}
    for path in TARGET.iterdir():
        if path.name not in keep and path.suffix == ".ipynb":
            path.unlink()

    readme = """# Ноутбуки дослідження RUL

Файли пронумеровано у рекомендованому порядку виконання. Етапи 01–06 зберігають стиль лабораторного журналу, 07–08 звужують пошук, а 09–13 оформлені як фінальна відтворювана частина.

| № | Ноутбук | Роль |
|---:|---|---|
"""
    roles = [
        "Підготовка спільних даних", "Базові моделі й відбір ознак", "Перевірка wavelet-ознак",
        "Власний CTO", "Ознаки початку деградації", "Сумісна перевірка груп ознак",
        "Пошук LSTM", "Пошук 1D-CNN", "Official test для w80", "Часткові історії",
        "Quantile q20", "Калібрування coverage", "Підсумкова таблиця і вибір",
    ]
    names = [item[1] for item in ORDER] + [final_path.name]
    for index, (name, role) in enumerate(zip(names, roles), start=1):
        readme += f"| {index:02d} | `{name}` | {role} |\n"
    readme += "\n`archive/` містить лише застарілу чернетку, яку не треба виконувати.\n"
    (TARGET / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
