from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs"
PDF_PATH = OUT_DIR / "rul_fd001_research_report_uk.pdf"

NAVY = colors.HexColor("#18344A")
BLUE = colors.HexColor("#2E6F95")
LIGHT_BLUE = colors.HexColor("#EAF2F7")
ORANGE = colors.HexColor("#C56B36")
LIGHT_ORANGE = colors.HexColor("#FAEFE8")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#5E6C78")
GRID = colors.HexColor("#CBD5DC")
WHITE = colors.white


def register_fonts() -> None:
    candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("C:/Windows/Fonts/calibri.ttf"), Path("C:/Windows/Fonts/calibrib.ttf")),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("Body", str(regular)))
            pdfmetrics.registerFont(TTFont("Body-Bold", str(bold)))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold")
            return
    raise FileNotFoundError("A Cyrillic-capable TrueType font was not found.")


class AccentLine(Flowable):
    def __init__(self, width: float, color=ORANGE, height: float = 3):
        super().__init__()
        self.width = width
        self.height = height
        self.color = color

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height, 1.5, fill=1, stroke=0)


def footer(canvas, doc) -> None:
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(GRID)
        canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
        canvas.setFont("Body", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 10 * mm, "Прогнозування RUL · NASA C-MAPSS FD001")
        canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"{page}")
    canvas.restoreState()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title", parent=base["Title"], fontName="Body-Bold", fontSize=28,
            leading=33, textColor=WHITE, alignment=TA_LEFT, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontName="Body", fontSize=12,
            leading=18, textColor=colors.HexColor("#D9E7F0"), spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Body-Bold", fontSize=19,
            leading=24, textColor=NAVY, spaceBefore=5, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Body-Bold", fontSize=13,
            leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Body", fontSize=10.2,
            leading=15.2, textColor=INK, spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Body", fontSize=8.3,
            leading=11.5, textColor=MUTED, spaceAfter=5,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Body", fontSize=10.2,
            leading=15, textColor=NAVY, backColor=LIGHT_BLUE, borderColor=BLUE,
            borderWidth=0.8, borderPadding=9, spaceBefore=5, spaceAfter=10,
        ),
        "warning": ParagraphStyle(
            "Warning", parent=base["BodyText"], fontName="Body", fontSize=10.2,
            leading=15, textColor=INK, backColor=LIGHT_ORANGE, borderColor=ORANGE,
            borderWidth=0.8, borderPadding=9, spaceBefore=5, spaceAfter=10,
        ),
        "caption": ParagraphStyle(
            "Caption", parent=base["BodyText"], fontName="Body", fontSize=8.2,
            leading=11, textColor=MUTED, alignment=TA_CENTER, spaceBefore=4, spaceAfter=9,
        ),
        "table": ParagraphStyle(
            "Table", parent=base["BodyText"], fontName="Body", fontSize=8.1,
            leading=10, textColor=INK,
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName="Body-Bold", fontSize=8.1,
            leading=10, textColor=WHITE, alignment=TA_CENTER,
        ),
    }


def P(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"• {text}", style)


def make_table(data: list[list[str]], widths: list[float], st: dict[str, ParagraphStyle]) -> Table:
    formatted = []
    for row_index, row in enumerate(data):
        style = st["table_head"] if row_index == 0 else st["table"]
        formatted.append([P(str(value), style) for value in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, colors.HexColor("#F5F8FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def make_charts() -> tuple[Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    raw = pd.DataFrame({
        "model": ["q20 LSTM", "LSTM w80", "Partial LSTM", "XGBoost", "CNN w80", "Partial CNN"],
        "MAE": [9.913, 10.430, 13.118, 15.571, 15.884, 17.661],
    }).sort_values("MAE", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    colors_list = ["#C56B36" if name == "q20 LSTM" else "#2E6F95" for name in raw["model"]]
    ax.barh(raw["model"], raw["MAE"], color=colors_list)
    for i, value in enumerate(raw["MAE"]):
        ax.text(value + 0.15, i, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlabel("MAE, цикли (менше — краще)")
    ax.set_title("Raw official test: точність моделей")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 19.5)
    fig.tight_layout()
    mae_path = TMP_DIR / "raw_mae_comparison.png"
    fig.savefig(mae_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    safety = pd.DataFrame({
        "metric": ["Overestimation, %", "Coverage, %", "NASA / 10"],
        "Base q20": [30.0, 70.0, 78.31],
        "Calibrated q20": [24.0, 76.0, 70.36],
    })
    x = range(len(safety))
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.bar([i - 0.18 for i in x], safety["Base q20"], width=0.36, label="Base q20", color="#9FB7C8")
    ax.bar([i + 0.18 for i in x], safety["Calibrated q20"], width=0.36, label="Calibrated q20", color="#C56B36")
    ax.set_xticks(list(x), safety["metric"])
    ax.set_title("Що змінило калібрування")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    safety_path = TMP_DIR / "calibration_tradeoff.png"
    fig.savefig(safety_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return mae_path, safety_path


def build() -> Path:
    register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mae_chart, safety_chart = make_charts()
    st = styles()

    doc = BaseDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
        title="Дослідження прогнозування RUL на NASA C-MAPSS FD001",
        author="Студент 2 курсу",
        subject="Feature engineering, LSTM, partial histories, q20 and calibration",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=footer)])

    story = []
    # Cover
    story += [Spacer(1, 24 * mm)]
    cover = Table([[P("ДОСЛІДЖЕННЯ ПРОГНОЗУВАННЯ RUL", st["title"]),
                    P("NASA C-MAPSS FD001", st["subtitle"])]],
                  colWidths=[115 * mm, 45 * mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 16 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16 * mm),
    ]))
    story += [cover, Spacer(1, 8 * mm), AccentLine(55 * mm), Spacer(1, 9 * mm)]
    story += [P("Від пошуку ознак і власного Custom Temporal Operator до partial-history навчання, q20-обмеження та калібрування безпечної моделі.", st["h2"])]
    story += [Spacer(1, 16 * mm)]
    story += [P("Формат: студентська дослідницька доповідь від першої особи", st["body"])]
    story += [P("Авторський контекст: 2 курс, 21 рік · проєкт predictive maintenance", st["small"])]
    story += [Spacer(1, 40 * mm)]
    story += [P("14 листопада 2026 · Кишинів", st["small"]), PageBreak()]

    story += [P("Коротко про результат", st["h1"])]
    story += [P("У цій роботі я намагався передбачити remaining useful life (RUL) двигуна за його історією сенсорів. Я почав із табличних моделей і ручної інженерії ознак, потім перейшов до LSTM і 1D-CNN, перевірив проблему коротких історій і в кінці зробив консервативну LSTM з quantile loss q=0.20.", st["body"])]
    story += [P("<b>Головний результат.</b> На raw official test найкраща q20 LSTM отримала MAE 9.913, RMSE 15.060, R² 0.869 і NASA Score 466.107. В окремому каліброваному запуску safety margin зменшив частку завищень з 30% до 24% і покращив NASA Score з 783.084 до 703.643, але MAE став гіршим на 1.12 циклу.", st["callout"])]
    story += [P("Я не трактую capped-at-125 метрики як заміну raw результатам. Capping змінює саму задачу, тому такі числа наведені лише як додатковий прикладний зріз.", st["warning"])]
    story += [P("Що я вважаю найважливішим висновком", st["h2"])]
    story += [bullet("краща середня точність ще не означає безпечний прогноз: завищення RUL штрафується сильніше;", st["body"])]
    story += [bullet("валідація на повних 80-циклових вікнах приховувала слабкість на коротких історіях;", st["body"])]
    story += [bullet("власні ознаки можуть допомогти, але виграш треба порівнювати з доданою складністю;", st["body"])]
    story += [bullet("калібрування — це не «безкоштовне» покращення: воно купує coverage ціною MAE та більш негативного bias.", st["body"])]

    story += [PageBreak(), P("1. Постановка задачі та протокол", st["h1"])]
    story += [P("Датасет FD001 містить траєкторії двигунів до відмови. Для кожного циклу є operating settings і 21 sensor. У train RUL можна відновити як кількість циклів до кінця траєкторії. В official test кожна траєкторія обривається раніше, а істинний RUL останнього доступного циклу лежить в окремому файлі.", st["body"])]
    story += [P("Щоб не отримати leakage, я ділив дані <b>за unit_id</b>, а не випадковими рядками. Preprocessing — scaler, detector baseline та інші параметри — fit-ився тільки на training engines. Validation використовував для вибору ознак, гіперпараметрів і кількості епох; official test — для фінальної одноразової оцінки.", st["body"])]
    metric_table = [
        ["Метрика", "Що показує", "Як читати"],
        ["MAE", "середня абсолютна помилка", "менше краще"],
        ["RMSE", "сильніше реагує на великі помилки", "менше краще"],
        ["R²", "частка поясненої варіації", "більше краще"],
        ["NASA Score", "асиметричний штраф", "завищення карається сильніше"],
        ["Coverage", "частка prediction ≤ true RUL", "для q20 ціль близько 0.80"],
    ]
    story += [make_table(metric_table, [28 * mm, 72 * mm, 60 * mm], st)]
    story += [Spacer(1, 3 * mm), P("У цій роботі помилка визначена як prediction − true. Позитивна помилка означає, що модель завищила ресурс — для maintenance це небезпечніший випадок.", st["small"])]

    story += [PageBreak(), P("2. Пошук базових ознак", st["h1"])]
    story += [P("Спочатку я хотів зрозуміти, чи достатньо табличної моделі. Побудував базові сенсорні ознаки, rolling statistics, delta та перевірив Linear Regression, HistGradientBoosting і XGBoost. На validation базовий XGBoost мав MAE близько 23.07, RMSE 29.86 і R² 0.793.", st["body"])]
    story += [P("Цей етап був навмисно трохи хаотичним: я швидко перевіряв групи ознак, кореляції, clipping і комбінації. Це допомогло сформувати інтуїцію, але одночасно показало ризик підігнати рішення під один validation split. Тому пізніше я додав GroupKFold і чіткіші протоколи.", st["body"])]
    story += [P("Wavelet-ознаки", st["h2"])]
    story += [P("Wavelet decomposition мала дати локальні характеристики форми сигналу. Найкращий набір d2_std зменшив MAE з 23.0717 до 22.9140 — приблизно на 0.16 циклу, або 0.68%. Додавання всіх wavelet-ознак навпаки погіршило MAE до 23.3065.", st["body"])]
    story += [P("Мій висновок: слабкий позитивний сигнал є, але він не виправдовує велику кількість додаткових колонок. Я залишив wavelets як експеримент, а не як основу фінальної моделі.", st["callout"])]

    story += [P("3. Власна ознака: Custom Temporal Operator", st["h1"])]
    story += [P("У <i>custom</i> ноутбуці я описав власну ідею CTO. Її сенс — порівнювати очікуваний локальний рух сенсора з фактичним рухом, а потім накопичувати disagreement уздовж траєкторії. Я хотів отримати не просто поточне значення sensor_11, а інформацію про те, наскільки його поведінка відповідає очікуваній деградації.", st["body"])]
    cto_table = [
        ["Конфігурація", "Ознак", "MAE", "RMSE", "R²"],
        ["Baseline", "25", "23.072", "29.859", "0.793"],
        ["CTO v2, sensor_11", "32", "22.650", "29.395", "0.800"],
        ["Expected + disagreement", "41", "22.304", "29.113", "0.803"],
        ["Full CTO", "81", "22.348", "29.145", "0.803"],
    ]
    story += [make_table(cto_table, [58 * mm, 20 * mm, 27 * mm, 28 * mm, 22 * mm], st)]
    story += [Spacer(1, 3 * mm), P("Найкращий варіант покращив MAE приблизно на 0.77 циклу (близько 3.3%) без потреби використовувати повний набір з 81 ознаки. Для мене це був важливий результат: власна ознака дала вимірюваний виграш, але простіший варіант виявився кращим за максимальну складність.", st["body"])]

    story += [PageBreak(), P("4. Детектор деградації та композиція", st["h1"])]
    story += [P("Наступною гіпотезою був Degradation Detector. Для кожного двигуна перші baseline cycles задавали його нормальний режим. Далі я рахував напрямлені відхилення для sensor_4, sensor_7, sensor_11, sensor_12 і sensor_15, підтверджував перевищення протягом кількох циклів і формував score, threshold excess та trend since degradation.", st["body"])]
    story += [P("Це рішення було мотивоване предметною логікою: модель має розрізняти нормальну ранню ділянку і момент, коли відхилення стає стійким. Я окремо перевіряв ознаки детектора групами A–E, а потім зробив Compose з wavelets, detector і CTO.", st["body"])]
    story += [P("У збережених Compose і Degradation Detector ноутбуках немає виконаних фінальних таблиць. Тому я не приписую цим етапам числовий виграш у доповіді. Їхня головна користь — формалізація preprocessing та ідея directional degradation score, яку далі використала q20 LSTM.", st["warning"])]
    story += [P("Чому я не зупинився на XGBoost", st["h2"])]
    story += [P("Таблична модель бачить підготовлений зріз, але не вчить часову структуру так прямо, як sequence model. Після інженерії ознак я перейшов до LSTM і 1D-CNN, щоб модель сама працювала з вікном історії.", st["body"])]

    story += [P("5. LSTM, CNN і пастка повних вікон", st["h1"])]
    story += [P("На validation найкраще працювали вікна довжиною 80. Для CNN середнє по seed також показало, що w80 помітно кращий за w30 і w50. На official test перша LSTM дала MAE 10.430, тоді як XGBoost — 15.571, а CNN — 15.884.", st["body"])]
    story += [Image(str(mae_chart), width=160 * mm, height=73 * mm), P("Рисунок 1. Raw official-test MAE. q20 і partial-history результати походять із наступних етапів.", st["caption"])]
    story += [P("Проблема з'явилася, коли я спробував наблизити training до реального онлайн-сценарію. У partial-history dataset кожен префікс траєкторії став окремим прикладом, а короткі історії доповнювалися зліва. Validation MAE для LSTM стала 18.622; для історій 1–19 циклів MAE була 42.261, а для 80+ — лише 8.402. Це чітко показало: повне вікно робило задачу значно легшою і приховувало ранню невизначеність.", st["body"])]
    partial_table = [
        ["Модель / історія", "MAE", "RMSE", "R²"],
        ["LSTM, усі prefixes", "18.622", "29.148", "0.803"],
        ["LSTM, 1–19 cycles", "42.261", "52.943", "−0.597"],
        ["LSTM, 20–79 cycles", "32.343", "39.967", "0.211"],
        ["LSTM, 80+ cycles", "8.402", "13.825", "0.910"],
    ]
    story += [make_table(partial_table, [62 * mm, 30 * mm, 32 * mm, 30 * mm], st)]

    story += [P("6. Чому я перейшов до q20", st["h1"])]
    story += [P("MSE або MAE симетричні: помилка +20 і −20 має однакову вагу. Для predictive maintenance це не зовсім правильно. Якщо модель завищує RUL, сервіс може бути запланований надто пізно. NASA Score теж відображає цю асиметрію: позитивна помилка зростає як exp(error/10), негативна — як exp(−error/13).", st["body"])]
    story += [P("Тому я використав quantile loss з q=0.20. Теоретично модель цілиться в нижній умовний квантиль RUL: overestimation отримує більшу вагу, а очікуваний conservative coverage наближається до 80%. Це не просто «відняти константу» після прогнозу — сама модель навчається на іншій функції втрат.", st["body"])]
    q20_table = [
        ["Протокол", "MAE", "RMSE", "R²", "NASA", "Over", "Coverage"],
        ["q20 raw", "9.913", "15.060", "0.869", "466.107", "46%", "54%"],
        ["q20 capped 125", "7.259", "11.040", "0.924", "175.493", "43%", "57%"],
    ]
    story += [make_table(q20_table, [45 * mm, 21 * mm, 22 * mm, 19 * mm, 24 * mm, 20 * mm, 22 * mm], st)]
    story += [P("q20 дала найкращу raw точність серед збережених official-test результатів. Але coverage 54% була далеко від очікуваних 80%. Це означало, що nominal quantile ще не є добре відкаліброваним quantile на нових двигунах.", st["warning"])]

    story += [P("7. Калібрування q20", st["h1"])]
    story += [P("Щоб не підбирати safety margin на official test, я розділив 100 train engines на 64/16/20: training, early stopping і calibration. Після refit на 80 development engines я взяв one-sided quantile помилки на 20 untouched calibration engines і відняв отриманий margin від прогнозу.", st["body"])]
    calibration_table = [
        ["Модель", "MAE", "RMSE", "NASA", "Over", "Coverage", "Bias"],
        ["Retrained base q20", "11.677", "17.456", "783.084", "30%", "70%", "−2.031"],
        ["Calibrated q20", "12.801", "17.881", "703.643", "24%", "76%", "−4.372"],
    ]
    story += [make_table(calibration_table, [43 * mm, 22 * mm, 23 * mm, 25 * mm, 19 * mm, 23 * mm, 22 * mm], st)]
    story += [Image(str(safety_chart), width=160 * mm, height=73 * mm), P("Рисунок 2. Калібрування зменшило overestimation і NASA Score та збільшило coverage.", st["caption"])]
    story += [P("Рішення дало потрібний напрямок: coverage 76% потрапила в практичний коридор 75–85%, а overestimation стала 24%. Водночас MAE погіршилася на 1.124 циклу, а bias став більш негативним. Я трактую це як чіткий safety–accuracy trade-off, а не як абсолютну перемогу каліброваної моделі.", st["callout"])]

    story += [P("8. Фінальний вибір моделей", st["h1"])]
    story += [P("Я б не залишав один артефакт на всі випадки. Для проєкту логічно мати два режими:", st["body"])]
    story += [bullet("<b>основна q20 LSTM</b> — коли потрібна найкраща raw точність і низький NASA Score;", st["body"])]
    story += [bullet("<b>calibrated q20 LSTM</b> — коли важливіше зменшити частку завищень і тримати coverage близько до 0.80.", st["body"])]
    story += [P("У двох фінальних Colab-ноутбуках я додав опційну MLflow-обгортку саме для моделей. Якщо задано MLFLOW_TRACKING_URI, вона зберігає повну PyTorch-модель, параметри, фінальні метрики й список ознак. Якщо URI не задано, training та export працюють без MLflow.", st["body"])]
    story += [P("Практичний критерій вибору", st["h2"])]
    decision = [
        ["Умова", "Модель"],
        ["мінімізувати raw MAE / RMSE", "Conservative LSTM q20"],
        ["зменшити overestimation до ~25%", "Calibrated q20"],
        ["простий інтерпретований baseline", "XGBoost + CTO"],
        ["демонстрація sequence baseline", "LSTM w80"],
    ]
    story += [make_table(decision, [96 * mm, 64 * mm], st)]

    story += [P("9. Обмеження і наступні кроки", st["h1"])]
    story += [bullet("FD001 має один operating condition і один fault mode; цього мало для висновку про реальне обладнання.", st["body"])]
    story += [bullet("Деякі ранні результати обиралися на одному validation split; потрібні repeated group splits або GroupKFold.", st["body"])]
    story += [bullet("Порівняння q20 запусків не повністю apples-to-apples: калібрований варіант використовує split 64/16/20 і окремий refit.", st["body"])]
    story += [bullet("Потрібні bootstrap confidence intervals на 100 official-test engines і перевірка кількох seed.", st["body"])]
    story += [bullet("Наступний сильний експеримент — однаковий split для MSE-LSTM, q20-LSTM і calibrated q20 з однаковим preprocessing.", st["body"])]
    story += [P("Особистий підсумок", st["h2"])]
    story += [P("Для мене це дослідження було не стільки про пошук одного красивого числа, скільки про поступове виправлення постановки задачі. Спочатку я додавав ознаки й архітектури. Потім побачив, що протокол повних вікон занадто оптимістичний. Після цього змістив фокус на partial histories і ризик завищення RUL. У результаті фінальна модель стала не лише точнішою, а й більш зрозумілою з точки зору того, яку помилку я намагаюся уникати.", st["callout"])]

    story += [PageBreak(), P("Додаток. Карта ноутбуків", st["h1"])]
    notebook_rows = [["№", "Файл", "Призначення"]]
    entries = [
        ("01", "01_dataset_preparation.ipynb", "підготовка даних"),
        ("02", "02_feature_search_xgboost.ipynb", "baseline і пошук ознак"),
        ("03", "03_wavelet_features.ipynb", "wavelet-гіпотеза"),
        ("04", "04_custom_temporal_operator.ipynb", "власна CTO-ознака"),
        ("05", "05_degradation_detector.ipynb", "детектор початку деградації"),
        ("06", "06_composed_features.ipynb", "композиція груп ознак"),
        ("07", "07_lstm_experiments.ipynb", "пошук LSTM"),
        ("08", "08_cnn_experiments.ipynb", "пошук 1D-CNN"),
        ("09", "09_official_test_w80.ipynb", "official test для w80"),
        ("10", "10_partial_history_training.ipynb", "навчання на prefixes"),
        ("11", "11_conservative_lstm_q20.ipynb", "q20 LSTM + MLflow model logging"),
        ("12", "12_conservative_lstm_q20_calibrated.ipynb", "калібрування + MLflow models"),
        ("13", "13_final_model_comparison.ipynb", "підсумкове порівняння"),
    ]
    notebook_rows.extend(entries)
    story += [make_table(notebook_rows, [11 * mm, 91 * mm, 58 * mm], st)]
    story += [Spacer(1, 5 * mm), P("Числа в доповіді взяті зі збережених outputs відповідних ноутбуків. Якщо notebook не містив виконаного фінального output, я описував лише метод і не приписував йому результат.", st["small"])]

    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    print(build())
