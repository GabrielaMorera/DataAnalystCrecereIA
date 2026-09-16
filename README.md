# Humanos vs. IA en gestión de cartera

Análisis reproducible de 100 llamadas (50 humanas y 50 de IA) para comparar eficiencia, conducta conversacional y desenlaces observados. Todo el procesamiento de audio se ejecuta localmente.

## Entregables

- `report/reporte_ejecutivo.html`: reporte ejecutivo de dos páginas, autocontenido y listo para imprimir.
- `data/processed/analytic_base.csv`: una fila por llamada, sin audio ni transcripción.
- `data/processed/hypothesis_tests.csv`: efectos, IC95%, pruebas y p-valores con corrección de Holm.
- `data/processed/summary.json`: resumen para auditoría o integración.
- `src/01_transcribe.py`: transcripción local y reanudable.
- `src/02_analyze.py`: variables, análisis estadístico y generación del reporte.
- `src/03_validate.py`: controles de integridad, privacidad y formato.
- `docs/ANALYSIS_PLAN.md`: preguntas, hipótesis y decisiones metodológicas.

## Enfoque

1. Inventario y control de calidad de WAV.
2. Transcripción offline con `faster-whisper` y detección de voz.
3. Variables acústicas, de conducta y resultado mediante reglas explícitas.
4. Mann–Whitney para continuas, Fisher para binarias e IC95% bootstrap (5.000 réplicas).
5. Corrección de Holm en 12 contrastes y GLM binomial ajustado por duración/contacto.

La comparación es observacional. Sin identificadores de campaña, saldo, mora, intento o asignación aleatoria, las diferencias no deben interpretarse como causales.

## Resultados clave

- **No hay un ganador estadístico en efectividad:** Humanos 82% vs. IA 70% en el proxy (−12 pp IA − Humano; `p Holm = 1,000`).
- **Tampoco en compromiso:** Humanos 44% vs. IA 30% (`p Holm = 1,000`).
- **La IA ocupa más la llamada:** proporción de habla mediana 97,5% vs. 90,3% (`p Holm < 0,001`).
- **La IA habla más despacio:** 134,6 vs. 154,1 palabras/min de habla (`p Holm = 0,002`).
- **La IA usa más fraseo empático explícito:** 72% vs. 28% (`p Holm < 0,001`); esto no prueba adaptación o calidez.
- **La diferencia más marcada es la presión legal:** 86% en IA vs. 16% en humanos (`p Holm < 0,001`).
- **Duración observada:** 2,5 min IA vs. 2,9 min humanos; la diferencia no es concluyente (`p Holm = 1,000`).

## Reproducción

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python src/01_transcribe.py --data-root "C:\ruta\audios_prueba_data_analyst"
python src/02_analyze.py --data-root "C:\ruta\audios_prueba_data_analyst"
python src/03_validate.py
```

Estructura esperada de entrada:

```text
audios_prueba_data_analyst/
├── audios_humanos_censurados/*.wav
└── audios_ia_censurados/*.wav
```

## Privacidad

El `.gitignore` excluye audios, transcripciones y datos intermedios. La base publicada conserva únicamente variables derivadas y el UUID técnico de cada archivo.

## Limitaciones principales

- Etiquetas de desenlace construidas como proxies; requieren auditoría manual/CRM.
- Whisper-base puede degradarse con telefonía de 8 kHz, ruido o solapamiento.
- No hay diarización confiable de agente/cliente en audio mono mezclado.
- Posible confusión por mezcla de campañas y momentos de cobranza.
