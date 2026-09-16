"""Fail-fast checks for the published analytical artifacts."""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
base = pd.read_csv(PROJECT / "data" / "processed" / "analytic_base.csv")
tests = pd.read_csv(PROJECT / "data" / "processed" / "hypothesis_tests.csv")
report = (PROJECT / "report" / "reporte_ejecutivo.html").read_text(encoding="utf-8")

assert len(base) == 100, f"Se esperaban 100 llamadas; hay {len(base)}"
assert base.group.value_counts().to_dict() == {"Humano": 50, "IA": 50}
assert base.call_id.is_unique, "Hay call_id duplicados"
assert not base[["call_id", "group", "duration_s"]].isna().any().any()
private_columns = {"transcript", "transcript_local_only", "text", "raw_text", "audio_path"}
assert private_columns.isdisjoint({column.lower() for column in base.columns})
assert len(tests) == 12, f"Se esperaban 12 contrastes; hay {len(tests)}"
assert tests.p_holm.between(0, 1).all()
assert report.count('<section class="page">') == 2, "El reporte no contiene exactamente dos páginas"
assert "http://" not in report and "https://" not in report, "El HTML no es autocontenido"

print("OK: 100 llamadas, balance 50/50, 12 contrastes y reporte autocontenido de 2 páginas")
