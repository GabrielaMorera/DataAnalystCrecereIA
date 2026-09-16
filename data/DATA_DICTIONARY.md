# Diccionario de la base analítica

Una fila representa una llamada. Los audios y las transcripciones se excluyen del repositorio por privacidad.

| Campo | Tipo | Definición |
|---|---:|---|
| `call_id` | texto | UUID del archivo, sin extensión. |
| `group` | categoría | `Humano` o `IA`, según la carpeta de origen. |
| `duration_s`, `duration_min` | continuo | Duración total del WAV. |
| `speech_s`, `speech_ratio` | continuo | Tiempo cubierto por segmentos con voz y proporción sobre la duración. |
| `mean_pause_s`, `p90_pause_s` | continuo | Pausas entre segmentos detectados por VAD. |
| `word_count`, `words_per_min` | continuo | Volumen de palabras transcritas y velocidad sobre tiempo hablado. |
| `questions_per_100_words` | continuo | Signos de pregunta por cada 100 palabras. |
| `transcript_confidence` | continuo | Media de `exp(avg_logprob)` de Whisper; indicador técnico, no KPI de agente. |
| `direct_contact_proxy` | binaria | No se detectó tercero, número errado o persona distinta del titular. |
| `wrong_party_proxy` | binaria | Expresiones explícitas de tercero/no titular. |
| `objection_proxy` | binaria | Incapacidad, negativa, desconocimiento de deuda u otra objeción explícita. |
| `handling_proxy` | binaria | Objeción junto con empatía, explicación, alternativa o propuesta. |
| `empathy_proxy` | binaria | Presencia de expresiones explícitas de comprensión o escucha. |
| `legal_pressure_proxy` | binaria | Mención de embargo, judicialización, proceso legal o reporte negativo. |
| `options_proxy` | binaria | Mención de opción, acuerdo, descuento, alivio o cuotas. |
| `negotiation_proxy` | binaria | Contacto directo con al menos dos señales de contexto de pago. |
| `commitment_proxy` | binaria | Expresión explícita de intención/fecha/comprobante de pago y contacto directo. |
| `refusal_proxy` | binaria | Negativa explícita a reconocer o pagar. |
| `effective_proxy` | binaria | Contacto directo con negociación o compromiso, excluyendo rechazo sin manejo. |
| `structure_score_0_5` | entero | Suma de validación, motivo, negociación, opciones y cierre. |
| `payment_context_hits` | entero | Conteo de familias de expresiones asociadas a pago/fecha/monto. |
| `outcome_proxy` | categoría | Tercero, compromiso, negociación, objeción/rechazo o información/cierre. |
| `rms_dbfs` | continuo | Nivel RMS medio del audio en dBFS. |
| `energy_active_ratio` | continuo | Proporción de marcos de 20 ms por encima de umbral adaptativo. |
| `clipping_pct` | continuo | Porcentaje de muestras cercanas al límite de amplitud. |

## Advertencia de uso

Las variables con sufijo `_proxy` son etiquetas reproducibles basadas en reglas y transcripción automática. Sirven para exploración y priorización; antes de convertirse en KPI deben validarse contra CRM y una muestra etiquetada por humanos.
