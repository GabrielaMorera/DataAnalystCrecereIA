"""Build the call-level analytical base, tests, and two-page executive report."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import unicodedata
import wave
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "raw"
TRANSCRIPTS = PROJECT / "data" / "interim" / "transcripts.jsonl"
OUT = PROJECT

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402

SEED = 8716


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in value if not unicodedata.combining(char))


PATTERNS = {
    "wrong_party": [
        r"\bno soy (yo|el|la|ese|esa|se[nñ]or|se[nñ]ora)",
        r"\b(no|ni) (lo|la) conozco\b",
        r"\bno conozco (a|al|el|la)\b",
        r"\bnumero (equivocado|errado)\b",
        r"\bse equivoco\b",
        r"\bes mi (hermano|hermana|madre|padre|hijo|hija|familiar)\b",
        r"\bno vive (aqui|aca)\b",
        r"\bno es (el|la) titular\b",
        r"\bfalleci[doa]\b",
    ],
    "objection": [
        r"\bno puedo (pagar|cancelar|hacer el pago)\b",
        r"\bno tengo (plata|dinero|trabajo|empleo)\b",
        r"\b(desemplead|sin trabajo|calamidad|enfermedad)\w*\b",
        r"\bno voy a pagar\b",
        r"\bno (debo|reconozco|acepto)\b",
        r"\bya pagu[eé]\b",
        r"\b(interes|intereses|muy caro|demasiado)\b",
        r"\bno me interesa\b",
        r"\bno estoy de acuerdo\b",
    ],
    "handling": [
        r"\b(entiendo|comprendo|comprendemos)\b",
        r"\b(lamento|disculpe|perm[ií]tame)\b",
        r"\b(opcion|alternativa|solucion|alivio|beneficio)\w*\b",
        r"\b(podemos|podria|podemos revisar|le propongo)\b",
        r"\b(descuento|condon|negoci)\w*\b",
    ],
    "payment_context": [
        r"\b(pago|pagar|cancelar|abono|cuota|consign|transfer)\w*\b",
        r"\b(acuerdo|compromiso) de pago\b",
        r"\b(mil|mill[oó]n|pesos)\b",
        r"\b(fecha|quincena|fin de mes)\b",
    ],
    "commitment": [
        r"\bvoy a (hacer|realizar|efectuar|generar)? ?(el )?(pago|abono)\b",
        r"\b(puedo|podr[ií]a) (pagar|cancelar|abonar)\b",
        r"\b(realizare|realizar[ií]a|hare|har[ií]a|efectuare) el pago\b",
        r"\bme comprometo\b",
        r"\b(acuerdo|compromiso) de pago (formal|para|por)\b",
        r"\b(pago|pagar|cancelar).{0,30}\b(el|dia|fecha) (lunes|martes|miercoles|jueves|viernes|sabado|domingo|\d{1,2})\b",
        r"\b(el|para el) (lunes|martes|miercoles|jueves|viernes|sabado|domingo|\d{1,2}).{0,30}\b(pago|pagar|cancelar)\b",
        r"\benviar(e|a|ia) (el )?comprobante\b",
    ],
    "refusal": [
        r"\bno voy a pagar\b",
        r"\bno puedo pagar\b",
        r"\bno (debo|reconozco|acepto)\b",
        r"\bno me interesa\b",
    ],
    "empathy": [
        r"\b(entiendo|comprendo|lamento|disculpe|tranquil[oa])\b",
        r"\bgracias por (aclararlo|contarme|su tiempo|atender)\b",
        r"\bperm[ií]tame (ayudar|revisar|explicar|atender)\b",
    ],
    "legal_pressure": [
        r"\b(embargo|judicial|proceso legal|cobro jur[ií]dico|demanda)\w*\b",
        r"\breporte negativo\b",
    ],
    "options": [
        r"\b(opcion|alternativa|alivio|beneficio|descuento|condon|cuota|acuerdo)\w*\b",
    ],
    "closing": [
        r"\b(whatsapp|wazah|correo|mensaje|comprobante)\b",
        r"\b(que tenga|feliz|excelente) (dia|tarde|noche)\b",
        r"\bgracias por (atender|su tiempo|la atencion)\b",
    ],
    "identity": [
        r"\b(confirma|confirmar|validar|verificar|titular|identidad)\w*\b",
        r"\bcon (el|la|don|do[nñ]a|se[nñ]or|se[nñ]ora)\b",
    ],
    "reason": [
        r"\b(motivo|razon) de (mi|la) llamada\b",
        r"\b(le llamo|me comunico|nos comunicamos) (por|para)\b",
        r"\b(obligacion|deuda|cartera|mora)\b",
    ],
}


def hits(text: str, key: str) -> int:
    return sum(len(re.findall(pattern, text)) for pattern in PATTERNS[key])


def merged_speech(segments: list[dict]) -> tuple[float, list[float], float]:
    intervals = sorted(
        (max(0.0, float(segment["start"])), max(0.0, float(segment["end"])))
        for segment in segments
        if float(segment["end"]) > float(segment["start"])
    )
    if not intervals:
        return 0.0, [], 0.0
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    speech = sum(end - start for start, end in merged)
    gaps = [merged[i][0] - merged[i - 1][1] for i in range(1, len(merged))]
    mean_segment = np.mean([end - start for start, end in intervals])
    return float(speech), gaps, float(mean_segment)


def acoustic_features(path: Path) -> dict[str, float]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        return {"rms_dbfs": np.nan, "energy_active_ratio": np.nan, "clipping_pct": np.nan}
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    normalized = samples / 32768.0
    rms = float(np.sqrt(np.mean(normalized**2)) + 1e-12)
    rms_dbfs = 20 * math.log10(rms)
    clipping_pct = float(np.mean(np.abs(samples) >= 32760) * 100)
    frame = max(1, int(rate * 0.02))
    usable = len(normalized) // frame * frame
    if usable:
        blocks = normalized[:usable].reshape(-1, frame)
        frame_rms = np.sqrt(np.mean(blocks**2, axis=1)) + 1e-12
        db = 20 * np.log10(frame_rms)
        noise_floor = float(np.percentile(db, 20))
        threshold = min(-28.0, max(-45.0, noise_floor + 8.0))
        active_ratio = float(np.mean(db > threshold))
    else:
        active_ratio = np.nan
    return {
        "rms_dbfs": round(rms_dbfs, 3),
        "energy_active_ratio": round(active_ratio, 4),
        "clipping_pct": round(clipping_pct, 5),
    }


def find_audio(group: str, filename: str) -> Path:
    folder = "audios_humanos_censurados" if group == "Humano" else "audios_ia_censurados"
    return DATA / folder / filename


def build_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        text_raw = record.get("text", "")
        text = normalize(text_raw)
        words = re.findall(r"\b[a-z0-9]+\b", text)
        word_count = len(words)
        duration = float(record["duration_s"])
        speech_s, gaps, mean_segment = merged_speech(record.get("segments", []))
        wrong = hits(text, "wrong_party") > 0
        objection = hits(text, "objection") > 0
        handling = hits(text, "handling") > 0
        pay_hits = hits(text, "payment_context")
        commitment = hits(text, "commitment") > 0 and not wrong
        negotiation = pay_hits >= 2 and not wrong
        refusal = hits(text, "refusal") > 0 and not wrong
        direct = not wrong
        effective = direct and (commitment or negotiation) and not (refusal and not handling)
        if wrong:
            outcome = "Tercero / no titular"
        elif commitment:
            outcome = "Compromiso"
        elif negotiation:
            outcome = "Negociación"
        elif refusal or objection:
            outcome = "Objeción / rechazo"
        else:
            outcome = "Información / cierre"
        structure = sum(
            [
                hits(text, "identity") > 0,
                hits(text, "reason") > 0,
                negotiation,
                hits(text, "options") > 0,
                hits(text, "closing") > 0,
            ]
        )
        confidence_values = [
            math.exp(float(segment.get("avg_logprob", -10)))
            for segment in record.get("segments", [])
        ]
        row = {
            "call_id": Path(record["file"]).stem,
            "group": record["group"],
            "duration_s": round(duration, 2),
            "duration_min": round(duration / 60, 4),
            "speech_s": round(speech_s, 2),
            "speech_ratio": round(speech_s / duration, 4) if duration else np.nan,
            "mean_pause_s": round(float(np.mean(gaps)), 3) if gaps else 0.0,
            "p90_pause_s": round(float(np.percentile(gaps, 90)), 3) if gaps else 0.0,
            "mean_segment_s": round(mean_segment, 3),
            "word_count": word_count,
            "words_per_min": round(word_count / (speech_s / 60), 2) if speech_s else np.nan,
            "questions_per_100_words": round(text_raw.count("?") / max(word_count, 1) * 100, 3),
            "transcript_confidence": round(float(np.mean(confidence_values)), 4)
            if confidence_values
            else np.nan,
            "direct_contact_proxy": int(direct),
            "wrong_party_proxy": int(wrong),
            "objection_proxy": int(objection),
            "handling_proxy": int(objection and handling),
            "empathy_proxy": int(hits(text, "empathy") > 0),
            "legal_pressure_proxy": int(hits(text, "legal_pressure") > 0),
            "options_proxy": int(hits(text, "options") > 0),
            "negotiation_proxy": int(negotiation),
            "commitment_proxy": int(commitment),
            "refusal_proxy": int(refusal),
            "effective_proxy": int(effective),
            "structure_score_0_5": int(structure),
            "payment_context_hits": int(pay_hits),
            "outcome_proxy": outcome,
        }
        row.update(acoustic_features(find_audio(record["group"], record["file"])))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "call_id"]).reset_index(drop=True)


def bootstrap_diff(a, b, statistic, iterations=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = statistic(a) - statistic(b)
    values = np.empty(iterations)
    for index in range(iterations):
        values[index] = statistic(rng.choice(a, len(a), replace=True)) - statistic(
            rng.choice(b, len(b), replace=True)
        )
    low, high = np.percentile(values, [2.5, 97.5])
    return float(observed), float(low), float(high)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        candidate = (m - rank) * pvalues[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_tests(frame: pd.DataFrame) -> pd.DataFrame:
    continuous = {
        "duration_min": "Duración (min)",
        "speech_ratio": "Proporción de habla",
        "words_per_min": "Palabras/min de habla",
        "questions_per_100_words": "Preguntas/100 palabras",
        "structure_score_0_5": "Estructura (0–5)",
        "mean_pause_s": "Pausa media (s)",
    }
    binary = {
        "direct_contact_proxy": "Contacto directo",
        "objection_proxy": "Objeción detectada",
        "empathy_proxy": "Empatía explícita",
        "legal_pressure_proxy": "Presión legal",
        "effective_proxy": "Efectividad proxy",
        "commitment_proxy": "Compromiso proxy",
    }
    rows = []
    ia = frame[frame.group == "IA"]
    human = frame[frame.group == "Humano"]
    for column, label in continuous.items():
        a = ia[column].dropna().to_numpy()
        b = human[column].dropna().to_numpy()
        result = stats.mannwhitneyu(a, b, alternative="two-sided")
        effect, low, high = bootstrap_diff(a, b, np.median)
        rows.append(
            {
                "metric": column,
                "label": label,
                "type": "continuous",
                "ia_value": float(np.median(a)),
                "human_value": float(np.median(b)),
                "effect_ia_minus_human": effect,
                "ci95_low": low,
                "ci95_high": high,
                "test": "Mann–Whitney U; bootstrap de medianas",
                "p_raw": float(result.pvalue),
            }
        )
    for column, label in binary.items():
        a = ia[column].to_numpy(dtype=float)
        b = human[column].to_numpy(dtype=float)
        table = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        result = stats.fisher_exact(table, alternative="two-sided")
        effect, low, high = bootstrap_diff(a, b, np.mean)
        rows.append(
            {
                "metric": column,
                "label": label,
                "type": "binary",
                "ia_value": float(np.mean(a)),
                "human_value": float(np.mean(b)),
                "effect_ia_minus_human": effect,
                "ci95_low": low,
                "ci95_high": high,
                "test": "Fisher exacta; bootstrap de proporciones",
                "p_raw": float(result.pvalue),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_raw.tolist())
    result["significant_5pct"] = result.p_holm < 0.05
    return result


def adjusted_model(frame: pd.DataFrame) -> dict:
    design = frame[["group", "duration_min", "direct_contact_proxy", "effective_proxy"]].copy()
    design["is_ia"] = (design.group == "IA").astype(int)
    design["log_duration"] = np.log1p(design.duration_min)
    x = sm.add_constant(design[["is_ia", "log_duration", "direct_contact_proxy"]])
    try:
        fit = sm.GLM(design.effective_proxy, x, family=sm.families.Binomial()).fit(cov_type="HC3")
        coefficient = float(fit.params["is_ia"])
        low, high = fit.conf_int().loc["is_ia"].tolist()
        return {
            "model": "GLM binomial: efectividad ~ IA + log(duración) + contacto directo",
            "odds_ratio_ia": math.exp(coefficient),
            "ci95_low": math.exp(float(low)),
            "ci95_high": math.exp(float(high)),
            "p_value": float(fit.pvalues["is_ia"]),
        }
    except Exception as error:
        return {"model": "GLM no estimable", "error": str(error)}


def pct(value: float, digits=0) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt(value: float, digits=1) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def grouped(frame: pd.DataFrame, column: str, statistic="mean") -> dict[str, float]:
    series = frame.groupby("group")[column]
    values = series.mean() if statistic == "mean" else series.median()
    return {str(key): float(value) for key, value in values.items()}


def metric_bars(label: str, human: float, ia: float, percent=True, maximum=1.0) -> str:
    def one(name, value, color):
        width = max(2.0, min(100.0, value / maximum * 100))
        shown = pct(value) if percent else fmt(value)
        return (
            f'<div class="bar-row"><span>{name}</span><div class="track">'
            f'<i style="width:{width:.1f}%;background:{color}"></i></div><b>{shown}</b></div>'
        )

    return (
        f'<div class="bar-group"><h4>{html.escape(label)}</h4>'
        + one("Humano", human, "#213555")
        + one("IA", ia, "#4ECDC4")
        + "</div>"
    )


def build_report(frame: pd.DataFrame, tests: pd.DataFrame, model: dict) -> str:
    duration = grouped(frame, "duration_min", "median")
    direct = grouped(frame, "direct_contact_proxy")
    effective = grouped(frame, "effective_proxy")
    commitment = grouped(frame, "commitment_proxy")
    objections = grouped(frame, "objection_proxy")
    empathy = grouped(frame, "empathy_proxy")
    legal = grouped(frame, "legal_pressure_proxy")
    questions = grouped(frame, "questions_per_100_words", "median")
    structure = grouped(frame, "structure_score_0_5", "median")

    objection_subset = frame[frame.objection_proxy == 1]
    handling = grouped(objection_subset, "handling_proxy") if len(objection_subset) else {"Humano": 0, "IA": 0}
    objection_n = objection_subset.group.value_counts().to_dict()
    for key in ["Humano", "IA"]:
        handling.setdefault(key, 0.0)
        objection_n.setdefault(key, 0)

    faster = "de IA" if duration["IA"] < duration["Humano"] else "humanas"
    efficiency = 1 - min(duration.values()) / max(duration.values())
    eff_winner = "IA" if effective["IA"] > effective["Humano"] else "Humanos"
    eff_gap = abs(effective["IA"] - effective["Humano"])
    significant = tests[tests.significant_5pct]
    duration_test = tests[tests.metric == "duration_min"].iloc[0]
    effective_test = tests[tests.metric == "effective_proxy"].iloc[0]

    if effective_test.p_holm < 0.05:
        headline = (
            f"Las llamadas {faster} duran {pct(efficiency)} menos; "
            f"{eff_winner} lidera la efectividad proxy por {eff_gap * 100:.0f} pp."
        )
    else:
        headline = (
            "La diferencia robusta está en el estilo, no en la efectividad: "
            f"{eff_winner} lleva {eff_gap * 100:.0f} pp observados, sin evidencia concluyente."
        )
    caution = (
        "Las etiquetas de resultado son proxies reproducibles sobre transcripción automática; "
        "deben auditarse antes de usarse como KPI operativo."
    )

    chart_outcomes = "".join(
        [
            metric_bars("Contacto directo", direct["Humano"], direct["IA"]),
            metric_bars("Negociación efectiva", effective["Humano"], effective["IA"]),
            metric_bars("Compromiso de pago", commitment["Humano"], commitment["IA"]),
            metric_bars("Objeción detectada", objections["Humano"], objections["IA"]),
        ]
    )
    chart_behavior = "".join(
        [
            metric_bars("Empatía explícita", empathy["Humano"], empathy["IA"]),
            metric_bars(
                f"Manejo de objeción (H n={objection_n['Humano']} · IA n={objection_n['IA']})",
                handling["Humano"],
                handling["IA"],
            ),
            metric_bars("Presión legal", legal["Humano"], legal["IA"]),
            metric_bars("Preguntas / 100 palabras", questions["Humano"], questions["IA"], False, max(1, max(questions.values()))),
        ]
    )

    findings = [
        f"La mediana dura {fmt(duration['IA'])} min en IA vs. {fmt(duration['Humano'])} min en humanos "
        f"(Δ {fmt(duration_test.effect_ia_minus_human)} min; IC95% {fmt(duration_test.ci95_low)} a {fmt(duration_test.ci95_high)}).",
        f"El contacto directo es {pct(direct['IA'])} en IA vs. {pct(direct['Humano'])} en humanos; "
        "la mezcla de terceros explica parte del embudo.",
        f"La efectividad proxy es {pct(effective['IA'])} en IA y {pct(effective['Humano'])} en humanos "
        f"(Δ {effective_test.effect_ia_minus_human * 100:+.0f} pp; p Holm={effective_test.p_holm:.3f}).",
        f"La IA usa presión legal en {pct(legal['IA'])} de llamadas vs. {pct(legal['Humano'])}; "
        "conviene probar guiones por segmento, no una cadencia única.",
    ]

    model_line = "Modelo ajustado no estimable."
    if "odds_ratio_ia" in model:
        model_line = (
            f"Al ajustar por duración y contacto directo, OR IA={model['odds_ratio_ia']:.2f} "
            f"(IC95% {model['ci95_low']:.2f}–{model['ci95_high']:.2f}; p={model['p_value']:.3f})."
        )

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Humanos vs. IA — Creceré AI</title>
<style>
@page{{size:A4;margin:0}}*{{box-sizing:border-box}}body{{margin:0;background:#e9edf2;color:#172033;font-family:Inter,Segoe UI,Arial,sans-serif}}
.page{{width:210mm;height:297mm;margin:14px auto;background:#fff;padding:12mm 13mm;position:relative;overflow:hidden;page-break-after:always}}
.page:last-child{{page-break-after:auto}}header{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #dce3ea;padding-bottom:4mm}}
.brand{{font-weight:900;letter-spacing:-.4px}}.brand em{{font-style:normal;color:#16a39a}}.tag{{font-size:9px;background:#e8faf8;color:#08756f;border-radius:20px;padding:5px 9px;font-weight:800;text-transform:uppercase;letter-spacing:.6px}}
h1{{font-size:27px;line-height:1.06;letter-spacing:-1px;margin:8mm 0 2mm;max-width:175mm}}.subtitle{{font-size:12px;color:#59657a;margin:0 0 6mm;max-width:175mm}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:3mm}}.kpi{{border:1px solid #dbe3ea;border-radius:10px;padding:3.5mm;background:#fbfcfd}}
.kpi b{{display:block;font-size:20px;letter-spacing:-.7px}}.kpi span{{font-size:8.5px;color:#687386;text-transform:uppercase;font-weight:800;letter-spacing:.4px}}.kpi small{{display:block;font-size:8px;color:#69758a;margin-top:2px}}
.grid{{display:grid;grid-template-columns:1.12fr .88fr;gap:6mm;margin-top:6mm}}.panel{{border:1px solid #dbe3ea;border-radius:12px;padding:5mm}}h2{{font-size:15px;margin:0 0 4mm;letter-spacing:-.2px}}h3{{font-size:11px;margin:0 0 2mm}}h4{{font-size:9.5px;margin:0 0 2px}}
.bar-group{{margin:0 0 3mm}}.bar-row{{display:grid;grid-template-columns:15mm 1fr 11mm;gap:2mm;align-items:center;font-size:8px;margin:2px 0}}.track{{height:7px;background:#edf1f4;border-radius:10px;overflow:hidden}}.track i{{height:100%;display:block;border-radius:10px}}.bar-row b{{font-size:8.5px;text-align:right}}
.findings{{padding:0;margin:0;list-style:none;counter-reset:item}}.findings li{{position:relative;padding:0 0 3.2mm 8mm;font-size:9.5px;line-height:1.33}}.findings li:before{{counter-increment:item;content:counter(item);position:absolute;left:0;top:-1px;width:5mm;height:5mm;border-radius:50%;background:#213555;color:#fff;font-size:8px;font-weight:800;text-align:center;line-height:5mm}}
.callout{{background:#0f766e;color:white;border-radius:11px;padding:4mm 5mm;font-size:9px;line-height:1.35;margin-top:3mm}}.callout b{{display:block;font-size:11px;margin-bottom:1mm}}
.footer{{position:absolute;bottom:7mm;left:13mm;right:13mm;display:flex;justify-content:space-between;color:#7b8596;font-size:7.5px;border-top:1px solid #e3e8ed;padding-top:2mm}}
.section-title{{display:flex;align-items:flex-end;justify-content:space-between;margin:7mm 0 4mm}}.section-title h1{{margin:0;font-size:23px}}.section-title p{{font-size:9px;color:#667085;margin:0;text-align:right;max-width:65mm}}
.method-grid{{display:grid;grid-template-columns:1fr 1fr;gap:4mm;margin-top:5mm}}.method{{border:1px solid #dbe3ea;border-radius:10px;padding:4mm}}.method p,.method li{{font-size:8.5px;line-height:1.35;color:#354052}}.method ul{{padding-left:4mm;margin:2mm 0 0}}
.actions{{display:grid;grid-template-columns:repeat(3,1fr);gap:3mm;margin-top:4mm}}.action{{background:#f2f6f8;border-radius:10px;padding:4mm;border-top:3px solid #4ECDC4}}.action b{{font-size:10px}}.action p{{font-size:8.3px;line-height:1.32;margin:2mm 0 0;color:#405066}}
.note{{font-size:7.6px;line-height:1.3;color:#69758a;margin-top:3mm}}.sig{{color:#0f766e;font-weight:800}}
@media print{{body{{background:white}}.page{{margin:0;box-shadow:none}}}}@media(max-width:850px){{.page{{width:100%;height:auto;min-height:297mm;margin:0}}}}
</style></head><body>
<section class="page"><header><div class="brand">CRECERÉ <em>AI</em></div><div class="tag">100 llamadas · análisis local</div></header>
<h1>{html.escape(headline)}</h1><p class="subtitle">Comparación exploratoria de 50 llamadas humanas y 50 de IA. Resultados observacionales; la asignación a campañas no está disponible.</p>
<div class="kpis">
<div class="kpi"><span>Duración mediana</span><b>{fmt(duration['IA'])} vs {fmt(duration['Humano'])}</b><small>min · IA vs Humano</small></div>
<div class="kpi"><span>Contacto directo</span><b>{pct(direct['IA'])} vs {pct(direct['Humano'])}</b><small>proxy · IA vs Humano</small></div>
<div class="kpi"><span>Efectividad</span><b>{pct(effective['IA'])} vs {pct(effective['Humano'])}</b><small>proxy · IA vs Humano</small></div>
<div class="kpi"><span>Compromiso</span><b>{pct(commitment['IA'])} vs {pct(commitment['Humano'])}</b><small>proxy · IA vs Humano</small></div>
</div>
<div class="grid"><div class="panel"><h2>Embudo y desenlace</h2>{chart_outcomes}<div class="callout"><b>Lectura ejecutiva</b>{html.escape(caution)}</div></div>
<div class="panel"><h2>Hallazgos que mueven decisiones</h2><ol class="findings">{''.join(f'<li>{html.escape(item)}</li>' for item in findings)}</ol></div></div>
<div class="footer"><span>Fuente: WAV entregados · Whisper-base local · reglas auditables</span><span>Página 1 / 2</span></div></section>

<section class="page"><header><div class="brand">CRECERÉ <em>AI</em></div><div class="tag">Conducta · método · acción</div></header>
<div class="section-title"><h1>La mejora está en el guion y la medición</h1><p>{len(significant)} de {len(tests)} contrastes siguen significativos tras corrección de Holm (5%).</p></div>
<div class="grid"><div class="panel"><h2>Conductas observadas</h2>{chart_behavior}</div><div class="panel"><h2>Qué explica la brecha</h2>
<p style="font-size:10px;line-height:1.4;margin:0 0 3mm">{html.escape(model_line)}</p>
<div class="callout"><b>No confundir correlación con causalidad</b>Sin campaña, saldo, edad de mora ni estrategia de marcación no es posible atribuir la brecha únicamente al tipo de agente.</div>
<p class="note">Estructura mediana (0–5): IA {fmt(structure['IA'])}; Humano {fmt(structure['Humano'])}. “Empatía” mide fraseo explícito, no calidez ni adaptación. Componentes de estructura: validación, motivo, negociación, opciones y cierre.</p></div></div>
<div class="method-grid"><div class="method"><h3>Preguntas e hipótesis</h3><ul><li>¿Quién logra contacto, negociación y compromiso con menor tiempo?</li><li>H1: IA exhibe menor dispersión y mayor estructura.</li><li>H2: humanos muestran más adaptación ante objeciones.</li><li>H3: la brecha persiste al controlar duración/contacto.</li></ul></div>
<div class="method"><h3>Cómo se contrastó</h3><ul><li>WAV → transcripción local + VAD + variables acústicas.</li><li>Mann–Whitney / Fisher; IC95% bootstrap (5.000).</li><li>12 contrastes con ajuste de Holm; GLM binomial robusto.</li><li>Auditar etiquetas proxy y estratificar por campaña antes de producción.</li></ul></div></div>
<div class="actions"><div class="action"><b>1 · Instrumentar</b><p>Guardar campaña, saldo, mora, intento, titularidad y desenlace validado. Es el mínimo para causalidad.</p></div>
<div class="action"><b>2 · Probar</b><p>A/B de guion: empatía + 2 alternativas + confirmación de fecha. Aleatorizar dentro de campaña.</p></div>
<div class="action"><b>3 · Escalar</b><p>IA para volumen y triage; transferencia humana cuando aparece objeción compleja o intención real de pago.</p></div></div>
<p class="note"><span class="sig">Definición de efectividad proxy:</span> contacto directo con negociación o compromiso, excluyendo rechazo sin manejo. Etiquetas derivadas de expresiones explícitas; no sustituyen CRM ni auditoría humana. Transcripciones y audios no se publican por privacidad.</p>
<div class="footer"><span>Reproducible: base analítica + pruebas + generador HTML</span><span>Página 2 / 2</span></div></section>
</body></html>"""


def main() -> None:
    global DATA, TRANSCRIPTS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--transcripts",
        type=Path,
        default=PROJECT / "data" / "interim" / "transcripts.jsonl",
    )
    parser.add_argument("--out", type=Path, default=PROJECT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    DATA = args.data_root.resolve()
    TRANSCRIPTS = args.transcripts.resolve()
    OUT = args.out.resolve()
    records = [json.loads(line) for line in TRANSCRIPTS.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 100 and not args.allow_partial:
        raise SystemExit(f"Expected 100 transcripts, found {len(records)}")

    processed = OUT / "data" / "processed"
    report_dir = OUT / "report"
    processed.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    frame = build_rows(records)
    tests = run_tests(frame)
    model = adjusted_model(frame)
    frame.to_csv(processed / "analytic_base.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(processed / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": int(len(frame)),
        "counts": frame.group.value_counts().to_dict(),
        "group_means": frame.groupby("group").mean(numeric_only=True).round(4).to_dict(orient="index"),
        "group_medians": frame.groupby("group").median(numeric_only=True).round(4).to_dict(orient="index"),
        "outcomes": pd.crosstab(frame.group, frame.outcome_proxy, normalize="index").round(4).to_dict(orient="index"),
        "adjusted_model": model,
    }
    (processed / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "reporte_ejecutivo.html").write_text(build_report(frame, tests, model), encoding="utf-8")

    # Local-only review file with transcripts and automated labels. It is deliberately outside the public deliverable.
    audit = frame[["call_id", "group", "outcome_proxy", "effective_proxy", "commitment_proxy"]].copy()
    text_map = {Path(record["file"]).stem: record["text"] for record in records}
    audit["transcript_local_only"] = audit.call_id.map(text_map)
    audit_path = OUT / "data" / "interim" / "audit_all_local_only.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    print(f"Built {len(frame)} rows in {OUT}")
    print(tests[["label", "ia_value", "human_value", "effect_ia_minus_human", "p_holm"]].to_string(index=False))
    print(json.dumps(model, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
