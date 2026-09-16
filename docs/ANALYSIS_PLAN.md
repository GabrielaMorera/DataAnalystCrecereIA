# Plan analítico y decisiones

## Preguntas

1. ¿Qué grupo logra más contacto directo, negociación y compromiso de pago?
2. ¿Qué costo en tiempo requiere cada desenlace?
3. ¿Qué conductas distinguen a Humanos e IA: preguntas, empatía, manejo, opciones y presión legal?
4. ¿La brecha de efectividad observada se mantiene al controlar duración y contacto directo?

## Hipótesis previas

- **H1 — Consistencia:** la IA tendrá mayor proporción de habla y estructura, con llamadas más homogéneas.
- **H2 — Adaptación:** los humanos mostrarán más manejo contextual de objeciones y compromisos explícitos.
- **H3 — Escala:** la IA logrará menor duración, aunque no necesariamente mejor conversión.
- **H4 — Confusión:** parte de la brecha estará asociada a contacto con terceros y mezcla de campañas.

## Construcción de variables

Se separaron tres capas para no llamar “efectiva” a una llamada solo por ser corta:

1. **Eficiencia/calidad técnica:** duración, cobertura de habla, pausas, velocidad, energía y clipping.
2. **Conducta:** preguntas, empatía, objeciones, manejo, alternativas, presión legal y estructura del guion.
3. **Desenlace:** tercero, contacto directo, negociación, compromiso, rechazo e información/cierre.

Las etiquetas de contenido usan familias de expresiones normalizadas, documentadas directamente en `src/02_analyze.py`. La evidencia privada permanece en `data/interim/` y no se publica.

## Contrastes

- Continuas: Mann–Whitney U; efecto como diferencia IA − Humano en medianas; IC95% bootstrap.
- Binarias: Fisher exacta; efecto como diferencia IA − Humano en proporciones; IC95% bootstrap.
- Multiplicidad: ajuste de Holm sobre 12 pruebas.
- Ajuste: GLM binomial `efectividad ~ IA + log(duración) + contacto directo`, con errores robustos HC3.

## Decisiones y límites

- No se infiere causalidad: falta campaña, saldo, edad de mora, intento, franja y asignación aleatoria.
- No se usa diarización: los WAV son mono mezclado; las métricas se calculan a nivel llamada.
- Se usa Whisper-base por equilibrio entre tiempo, reproducibilidad y privacidad local.
- Los proxies deben contrastarse con CRM y una muestra auditada antes de operar incentivos o decisiones sobre agentes.
