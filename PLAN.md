# Motor de asignación y riesgo con saltos — Plan de ejecución

**Objetivo:** demostrar dominio de cálculo estocástico, optimización convexa, VaR/CVaR y
validación de modelos de riesgo.

**La pregunta que ordena todo el proyecto:**

> ## ¿Cuánto capital te cuesta el modelo de riesgo equivocado?

**Dos actos:**

| | Contenido | Días | Naturaleza |
|---|---|---|---|
| **Acto 1** | Modelas la realidad (Merton) y asignas capital (mín-CVaR) | 1-10 | Núcleo — entregable completo por sí solo |
| **Acto 2** | Auditas 9 especificaciones de VaR/ES y traduces cada veredicto a capital regulatorio vía Basilea | 11-13 | Aditivo — no reescribe nada del Acto 1 |

**Tesis del Acto 1:** el mundo tiene saltos. Tanto la asignación de capital como la medición
de riesgo cambian cuando dejas de fingir que no.

**Tesis del Acto 2:** medir mal el riesgo no es un error académico. Tiene un precio en capital
regulatorio, y se puede calcular.

**Restricciones duras:**
- 10 días hábiles para el Acto 1. El Acto 2 son 3 más y puede posponerse sin costo.
- Datos gratis e instantáneos. Nada que requiera recolección prolongada ni API de pago.
- Un solo modelo estocástico generador (Merton). Un solo nivel de confianza principal (99%).

---

## ESTADO: Acto 1 COMPLETO ✅ (días 1-10)

3,923 días out-of-sample, 188 rebalanceos, 4 carteras × 5 modelos, 16 checks pasando.

| Tesis | Veredicto |
|---|---|
| 1. El VaR gaussiano subestima la cola | ✅ **Confirmada** — 2.06× las excepciones esperadas, reprueba Kupiec en las 4 carteras. Merton 0.97× y pasa en las 4 |
| 2. Mín-CVaR reduce la cola realizada | ❌ **Negativa** — asigna 10 pp distinto (señal real, 5.5× el ruido) pero CVaR realizado +0.1% y drawdown 4.8% peor |
| 3. El modelo tiene precio en capital | ⚠️ **Confirmada con el signo invertido** — el modelo que reprueba el backtest cuesta **18.4% MENOS** capital que el que lo pasa |

### Acto 2 (días 11-12) ✅ — 10 modelos × 4 carteras

**Pregunta del Acto 2 — ¿condicionar por volatilidad arregla la independencia?
Respuesta: reduce el agrupamiento 4-8× pero no lo elimina.**

Persistencia π_11/π_01 (1.0 = sin agrupamiento): `ewma` 2.3-4.6 y `fhs` 4.4-9.8, contra
`mc_merton` 15.9-30.1 y `cornish_fisher` 17.4-39.4. De 40 celdas, **una sola pasa
independencia**: `igual_peso/ewma` (p=0.146).

**El intercambio es nítido y ningún modelo lo resuelve solo:**

| | Frecuencia (Kupiec) | Momento (independencia) | Por qué |
|---|---|---|---|
| `ewma` | **peor** (2.14×, reprueba 4/4) | **mejor** (π 2.3) | vol condicional, colas gaussianas |
| `mc_merton` | **mejor** (0.97×, pasa 4/4) | casi el peor (π 16-30) | colas gordas, vol plana |
| `fhs` | buena (1.07-1.35, pasa 3/4) | 2ª mejor (π 4.4) | **las dos cosas** |

FHS es la síntesis: volatilidad condicional de EWMA + forma empírica de la cola. Es la única
decente en ambos ejes, y es la razón de que gane las comparativas publicadas.

**Acerbi-Székely (ES):** `normal`, `mc_gbm` y `ewma` reprueban en las 4 carteras con Z2 entre
−1.1 y −1.6. `mc_merton`, `t_student`, `cornish_fisher` y `evt` pasan en las 4.

### El hallazgo que invierte la tesis 3

Capital = multiplicador × VaR medio, sobre $10M nocional, cartera `min_var`:

```
mc_gbm         $389,062   +0.0%   ← REPRUEBA Kupiec, CC y ES
normal         $389,877   +0.2%   ← REPRUEBA todo
ewma           $393,288   +1.1%   ← REPRUEBA Kupiec y ES
fhs            $427,270   +9.8%
mc_merton      $460,524  +18.4%   ← PASA todo
cornish_fisher $530,557  +36.4%
```

**El modelo que reprueba el backtest es el más barato.** El mecanismo: el multiplicador castiga
como máximo +33% (3.00 → 4.00), pero subestimar el VaR ahorra ~20% de forma directa. Y el
castigo real es mucho menor que el tope: con 2× la tasa de excepciones, un modelo promedia
5.2 excepciones por ventana de 250 días — apenas la primera banda amarilla. **El semáforo tiene
poca potencia en ventanas de 250 días**, y por eso no alcanza a corregir el incentivo.

Esto no es un error del cálculo: es la crítica documentada al marco de backtesting de Basilea,
y una de las razones por las que Basilea III (FRTB) migró la métrica de capital de VaR a ES y
mantiene requisitos cualitativos de aprobación además del test cuantitativo.

Mi hipótesis original —que reprobar el backtest encarecería el capital— era incorrecta en el
signo. Se reporta como salió.

**Hallazgo no planeado, y el que mejor arma el Acto 2:** las cinco especificaciones reprueban
independencia. Tras una excepción la probabilidad se multiplica por 16.4×. Ninguna de las cinco
tiene volatilidad variable en el tiempo — y EWMA y FHS del día 11 sí. La pregunta del Acto 2
pasa a ser si condicionar por volatilidad arregla la independencia.

### Auditoría final ✅ — 6 hallazgos aplicados

| Severidad | Hallazgo | Corrección |
|---|---|---|
| **Alta** | El test de Acerbi-Székely **no tenía potencia**: rechazaba 4/100 con el ES 50% subestimado. Su nulo remuestreaba los cocientes de cola observados, contaminándolo con la alternativa | Nulo reconstruido desde el modelo (exceso ~ Exp(ES − VaR)). Potencia 1/60 → 12/60 → 36/60 según severidad. **Cambió veredictos y la recomendación del informe** |
| **Alta** | `data.load_prices()` devolvía el caché aunque `TICKERS` hubiera cambiado: todo el pipeline corría sobre un universo que nadie pidió | El caché se invalida si su universo no coincide |
| **Media** | 5 de 10 modelos compartían color en las figuras 5 y 6, las que sostienen las conclusiones | Paleta completa, 10 colores únicos. `fig1` conserva solo los 5 del Acto 1 para no saturarse |
| **Media** | `evt()` devolvía `var*1.5` cuando ξ ≥ 1 — un número inventado | ES empírico, que es finito y observable. Nunca se dispara (ξ real entre −0.17 y +0.53) |
| **Baja** | `mc_merton` degradaba a `mc_gbm` en silencio pese a que su docstring decía registrarlo | `warnings.warn`. Nunca se dispara, verificado sobre todas las ventanas |
| **Baja** | `pd_skew`/`pd_kurt` importaban pandas dentro de la función | `scipy.stats` con `bias=False`, idéntico al estimador de pandas (verificado) |

**Verificado y sin hallazgo:** determinismo (misma semilla ⇒ resultado idéntico en los 10
modelos), coherencia `ES ≥ VaR` (0 violaciones en 156,920 filas), y que ni la calibración de
Merton ni ξ ≥ 1 fallan en ninguna ventana real.

**Riesgo residual:** el p-valor del ES es simulado, así que un veredicto cerca de 0.05 lleva
ruido Monte Carlo. Con `n_boot=20 000` el rango entre semillas es 0.003 — suficiente para los
valores observados, pero un caso limítrofe futuro exigiría más réplicas.

### Registro de correcciones al plan

Errores míos en los criterios originales, detectados al ejecutarlos:

| Fase | Criterio original | Por qué estaba mal | Criterio real |
|---|---|---|---|
| 2-3 | "momentos simulados dentro del 10%" | EEM tiene skew 0.033: el error relativo daba 142% sin indicar nada | Partido en dos: cumulantes cerrados vs empíricos (rtol 1e-6), y simulador vs cumulantes (tolerancia **absoluta** en skew) |
| 4 | "el error decrece con N" | Comparar realizaciones sueltas y exigir monotonía es inválido; falla al azar | Sesgo < 4 errores estándar, y el SE escala como 1/√N |
| 6 | "mín-CVaR converge a mínima varianza" | CVaR incluye el término de media: es equivalencia **media-varianza**, no mínima varianza | Escenarios centrados, o misma restricción de retorno |
| 7 y 11 | "`ES ≤ VaR`" | Invertido. ES promedia las pérdidas *peores* que el VaR | **`ES ≥ VaR`** en convención de pérdida |

### Desviaciones de alcance (deliberadas)

- **Salto sistémico: hecho en el día 4**, no como opcional. La causa fue técnica: `Σ − diag(var_saltos)`
  sale no-PSD con activos correlacionados a 0.95. Y quedó justificado con una medida —
  ES/VaR de 1.159 con saltos independientes contra 1.162 del gaussiano: **las colas
  idiosincrásicas se diversifican, las sistémicas no**.
- **+1 cartera** (`igual_peso`) y **+1 modelo** (`mc_merton_idio`) como controles.
- **Caché de calibración** en `risk._calibrar_ventana`: el walk-forward pedía 64 calibraciones
  diarias donde bastan 8. De ~20 min a **3:06**. La mitigación de "bajar a rebalanceo
  trimestral" queda descartada: no hay problema de cómputo.

---

## Estado verificado (antes de escribir código)

| Ítem | Estado |
|---|---|
| Descarga `yfinance` de los 8 activos | ✅ 4,924 días × 8, 2007-01-03 → 2026-07-30, sin NaN |
| `numpy` 2.5.1, `pandas` 3.0.5, `scipy` 1.18.0, `matplotlib` 3.11.1, `yfinance` 1.5.2 | ✅ instalados |
| `cvxpy` | ❌ falta — única dependencia nueva |

Universo: `SPY QQQ IWM EFA EEM TLT GLD DBC`
Ventana: 2007-01-03 → hoy. Cubre 2008, 2011, 2015, 2018, 2020 y 2022.

---

## Arquitectura del repo

```
jump-risk-engine/
├── README.md              # entregable ejecutivo: supuestos → método → resultado → recomendación
├── PLAN.md                # este archivo
├── requirements.txt
├── data/
│   └── prices.csv         # caché local, se descarga una vez
├── src/
│   ├── data.py            # ingesta + caché + retornos log
│   ├── merton.py          # calibración y simulación del jump-diffusion
│   ├── optimize.py        # mín-CVaR (LP), Markowitz (QP), risk parity
│   ├── risk.py            # REGISTRO de modelos VaR/ES — ver Ruta 7
│   ├── backtest.py        # walk-forward + Kupiec + Christoffersen
│   ├── basel.py           # ── Acto 2: semáforo y multiplicador de capital
│   ├── evt.py             # ── Acto 2: peaks-over-threshold / GPD
│   └── plots.py           # figuras estáticas a PNG
├── tests/
│   └── test_core.py       # los checks de validación de cada fase
└── figures/
```

Sin notebooks en el repo final. Un `explore.ipynb` fuera de `src/` para trabajo sucio está
bien, pero no es el entregable.

---

# Las 7 rutas a decidir

### Ruta 1 — De dónde salen los escenarios del optimizador

| Opción | Ventaja | Costo |
|---|---|---|
| (a) Bootstrap histórico | Sin supuestos | Solo genera lo que ya pasó |
| (b) Simulación desde Merton calibrado | Genera colas nunca observadas | Depende del modelo |
| (c) Filtered bootstrap | Lo mejor de ambos | +2 días |

**→ Recomendado: (b) como principal, (a) como benchmark de contraste.**
La tesis del proyecto es que los saltos importan, y eso solo se ve si los escenarios los
contienen. Además, comparar (a) contra (b) *es* uno de los resultados: cuánto se mueven los
pesos óptimos al cambiar el generador de escenarios. Descartar (c) por presupuesto.

### Ruta 2 — Cómo calibrar Merton

| Opción | Tiempo | Riesgo |
|---|---|---|
| (a) MLE, verosimilitud de mezcla Poisson truncada a ~10 términos | 1.5 días | **Alto** — ver abajo |
| (b) Método de momentos: iguala varianza, skew y kurtosis | 3 horas | Bajo |

**→ Recomendado: implementa (b) primero, y solo entonces intenta (a).**

Razón no obvia: la verosimilitud de Merton es **no acotada**. Si dejas que un salto explique
exactamente una observación con varianza de salto tendiendo a cero, la verosimilitud diverge.
Es una patología conocida del modelo, no un bug tuyo, y puede quemarte un día entero.
Mitigación si vas por (a): acota `sigma_J` por abajo (p. ej. ≥ 0.005) y arranca el optimizador
desde los parámetros que te dio (b).

Tener (b) funcionando el día 3 te garantiza un pipeline end-to-end completo aunque (a) falle.
Esa es toda la razón del orden.

### Ruta 3 — Diseño del walk-forward

**→ Ventana rodante de 1,000 días (~4 años), reoptimización mensual.**

Rodante y no expansiva porque el régimen cambia y una ventana expansiva diluye 2020 dentro de
2008. Mensual da ~180 rebalanceos y deja suficientes días entre ellos para medir.

**Punto crítico de diseño:** el VaR se recalcula **diario**, la cartera se reoptimiza
**mensual**. Si reoptimizas diario, la cartera cambia debajo de la medición y el backtest de
VaR pierde significado — estarías midiendo el riesgo de una cartera que nunca existió un día
completo. Este error es común y descalifica el backtest entero.

Los primeros 1,000 días se consumen en la ventana inicial → backtest efectivo desde ~2011,
con ~3,900 observaciones. De sobra para Kupiec (necesitas ≥250) y suficiente para **15
ventanas Basilea independientes de 250 días** en el Acto 2.

### Ruta 4 — Qué es exactamente una "excepción"

Definición operativa, a fijar antes de escribir `backtest.py`:

> Con los pesos `w` del último rebalanceo, el VaR al 99% predicho al cierre del día *t* se
> compara contra el retorno realizado de la cartera en *t+1*. Hay excepción si la pérdida
> realizada supera el VaR predicho. Sin costos de transacción.

Bajo el modelo correcto, ~1% de los días son excepción. Kupiec prueba si la *frecuencia* es la
correcta; Christoffersen prueba si están *independientes* en el tiempo. Un modelo puede pasar
Kupiec y reprobar Christoffersen — las excepciones se apelmazan en crisis. Ese caso es el
resultado más interesante que puedes encontrar y hay que buscarlo explícitamente.

### Ruta 5 — Optimizador: `cvxpy` o `scipy`

**→ `cvxpy`.** El LP de Rockafellar-Uryasev son 6 líneas; a mano en `scipy.optimize.linprog`
son ~40 de ensamblado de matrices dispersas con variables auxiliares, y es donde más fácil se
cuela un error silencioso. Además `cvxpy` resuelve también el QP de Markowitz.

Fallback si la instalación falla: `scipy.optimize.linprog` con matriz dispersa. No es
bloqueante, solo más lento de escribir.

### Ruta 6 — Alcance del entregable

**→ El README es parte del proyecto, no el epílogo.** Dos páginas que un director lea sin
abrir código: supuestos → método → resultado → recomendación → limitaciones conocidas.
Presupuestado dentro del día 10, no "cuando termine".

### Ruta 7 — Registro de modelos ⚠️ *la decisión que habilita el Acto 2*

**Se toma el día 7. No es retrofitable sin dolor.**

`risk.py` **no** debe tener los métodos de VaR escritos a mano uno tras otro. Debe exponer un
**registro** donde cada especificación es una función con firma común:

```python
# firma única para toda especificación de riesgo
def modelo(retornos_ventana: np.ndarray, nivel: float) -> tuple[float, float]:
    """Devuelve (VaR, ES) para la ventana dada."""

REGISTRO = {
    "historico":   var_historico,
    "normal":      var_normal,
    "mc_gbm":      var_mc_gbm,
    "mc_merton":   var_mc_merton,
    # Acto 2 añade 5 entradas aquí y nada más
}
```

Y `backtest.py` itera sobre `REGISTRO`, no sobre nombres codificados.

**Por qué importa:** con el registro, el Acto 2 es agregar cinco funciones y cinco líneas a un
diccionario. Sin él, es una refactorización de todo el bucle de backtest. Mismo esfuerzo el
día 7; diferencia de días el día 11.

---

# ACTO 1 — Fases día por día

Cada fase tiene un check ejecutable. Si el check no pasa, no se avanza a la siguiente.

### Día 1 — `data.py`
Ingesta, caché a CSV local, retornos logarítmicos.

**Validación:** matriz 4,924 × 8 sin NaN; volatilidad anualizada de cada activo dentro de
rangos plausibles (TLT ~13%, SPY ~19%, EEM ~22%); la kurtosis empírica de todos los activos
es > 3 (si diera ≈3, hay un bug en el cálculo de retornos).

### Días 2-3 — `merton.py` (calibración)
Método de momentos primero. MLE después si hay tiempo.

**Validación:** simula 100,000 días con los parámetros calibrados y compara los momentos 3 y 4
contra los empíricos. Deben reproducirse dentro del ~10%. Este es el check que demuestra que
la calibración capturó las colas, que es el punto entero del modelo.

### Día 4 — `merton.py` (simulación)
Generador de escenarios: GBM y Merton con la misma deriva y difusión.

**Validación:** `E[S_T] → S_0·e^{μT}` con error estándar decreciente en √N. Es el mismo
control de convergencia que ya hiciste en `02_MC_Engine_Sim.ipynb`; reutiliza esa lógica.

### Días 5-6 — `optimize.py`
Mín-CVaR (LP), Markowitz (QP), risk parity.

**Validación — el check más importante del proyecto, dos partes:**

1. El `α` óptimo del LP debe coincidir con el percentil empírico de las pérdidas de la cartera
   óptima. Si no coincide, la formulación está mal.
2. **Alimenta el optimizador con escenarios gaussianos.** Bajo normalidad, CVaR es función
   monótona de la varianza, así que la cartera mín-CVaR *debe* converger a la de mínima
   varianza de Markowitz. Si no convergen, tu LP tiene un error. Este test es la razón de que
   valga la pena implementar Markowitz aunque no fuera parte de la tesis.

### Día 7 — `risk.py` — **construir el REGISTRO (Ruta 7)**
Cuatro especificaciones iniciales bajo la firma común: histórico, paramétrico normal, MC bajo
GBM, MC bajo Merton.

**Validación:** el VaR histórico al 99% debe cortar exactamente el 1% de las observaciones
in-sample — trivialmente cierto por construcción, y por eso sirve como sanity check del código
de conteo que después usará el backtest. Adicional: recorrer `REGISTRO` completo sobre una
ventana cualquiera debe devolver pares `(VaR, ES)` finitos, con `VaR > 0` y **`ES ≥ VaR`** en
convención de pérdida (el ES promedia las pérdidas peores que el VaR).

### Días 8-9 — `backtest.py`
Bucle walk-forward iterando sobre el registro. Reoptimizar mensual, recalcular VaR diario,
acumular excepciones por modelo.

**Es la fase que se come el tiempo. Presupuéstala completa y no la comprimas.**

**Validación — prueba el test antes de confiar en él:** genera 1,000 días sintéticos de
excepciones Bernoulli(0.01) y verifica que Kupiec **no** rechaza; repite con Bernoulli(0.05) y
verifica que **sí** rechaza. Si tu Kupiec no distingue esos dos casos, está mal implementado y
todos los veredictos del proyecto son ruido.

### Día 10 — `plots.py` + README
Figuras estáticas y entregable ejecutivo.

**Validación:** dale el README a alguien que no sepa de finanzas. Si no puede decirte cuál es
la recomendación, reescríbelo.

> **🚩 Línea de corte.** Aquí el proyecto está completo y es presentable. El Acto 2 se puede
> retomar después — un fin de semana, entre parciales — sin tocar nada de lo construido.

---

# ACTO 2 — Auditoría de modelos grado Basilea

Tres días. Cinco especificaciones nuevas y tres pruebas nuevas. Ninguna toca el bucle de
backtest: todas entran por el registro del día 7.

### Día 11 — Cinco especificaciones nuevas

| Modelo | Costo | Por qué está |
|---|---|---|
| **EWMA / RiskMetrics** (λ=0.94) | 1 h | El baseline de la industria. Captura clustering de volatilidad sin estimar nada |
| **Paramétrico t-Student** | 1-2 h | El contraste directo contra el normal: mismas colas, distinto grosor |
| **Cornish-Fisher** | 30 min | Usa skew y kurtosis que ya calculas en el día 1 |
| **FHS** (Filtered Historical Simulation) | ½ día | Filtra con EWMA, estandariza residuos, bootstrapea, reescala por vol actual. Suele ganar las comparativas publicadas |
| **EVT peaks-over-threshold** (`evt.py`) | 1 día | GPD sobre excedencias vía `scipy.stats.genpareto` |

Orden de implementación: los tres baratos primero (quedas en 7 modelos en media mañana), luego
FHS, luego EVT.

**EVT — no sobre-optimices el umbral.** Fíjalo en el percentil 95 de las pérdidas y
documéntalo como decisión. La selección "óptima" de umbral es un tema de tesis entero y no
mejora el resultado del proyecto.

**Validación:** los diez modelos deben producir VaR monótono en el nivel de confianza
(VaR 99% ≥ VaR 95% en pérdida) y **`ES ≥ VaR`**. Un modelo que viole eso tiene un bug de signo —
el error más común de esta fase.

### Día 12 — Tres pruebas nuevas

**Christoffersen condicional (CC)** — 30 min. Es la suma de las razones de verosimilitud de
POF e independencia, contrastada contra χ²(2). Prácticamente gratis si ya tienes las otras dos.

**Acerbi-Székely Test 2** — ½ día. Es lo que separa "calculé ES" de "validé ES". Backtest
incondicional del Expected Shortfall, con el p-valor por simulación bajo la hipótesis nula.
Relevante porque Basilea III (FRTB) migró la métrica de capital de VaR a ES.

**`basel.py` — semáforo y multiplicador** — 1 h. **La pieza de mayor retorno por hora de todo
el proyecto.**

Sobre ventanas de 250 días hábiles, con VaR al 99%:

| Zona | Excepciones | Multiplicador de capital |
|---|---|---|
| 🟢 Verde | 0 – 4 | 3.00 |
| 🟡 Amarilla | 5 | 3.40 |
| | 6 | 3.50 |
| | 7 | 3.65 |
| | 8 | 3.75 |
| | 9 | 3.85 |
| 🔴 Roja | 10 o más | 4.00 |

Es una tabla de consulta sobre un conteo. Veinte líneas de código. Y es lo que **convierte la
calidad estadística del modelo en un número en dólares**: el capital requerido escala con el
multiplicador, así que reprobar el backtest tiene un costo directo y calculable.

**Validación:** alimenta `basel.py` con conteos de 4, 5, 9 y 10 y verifica que devuelve
verde/amarillo/amarillo/rojo con los multiplicadores exactos de la tabla.

### Día 13 — Síntesis y README v2

La tabla maestra del proyecto: **9 modelos × 15 ventanas Basilea**, con excepciones, veredicto
de Kupiec, Christoffersen IND, Christoffersen CC, Acerbi-Székely, zona de semáforo y
multiplicador promedio.

Y la traducción final, que es la frase de consultor del proyecto:

> *"Sobre una posición de $10M, usar el VaR paramétrico normal en vez de FHS te manda a zona
> amarilla en N de 15 ventanas y eleva el requerimiento de capital en X% — $Y anuales."*

**Validación:** el README v2 debe responder la pregunta del encabezado — *¿cuánto capital te
cuesta el modelo equivocado?* — con una cifra, en la primera media página.

---

## Riesgos conocidos y mitigación

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| MLE de Merton no converge (verosimilitud no acotada) | Alta | Método de momentos como camino principal; MLE es mejora opcional |
| El walk-forward se desborda del presupuesto | Media | Vectorizar el bucle interno; reducir a reoptimización trimestral si aprieta |
| Errores de signo en los 9 modelos del Acto 2 | Media | El check de monotonía y `ES ≤ VaR` del día 11, corrido sobre todo el registro |
| `yfinance` con rate limit | Baja | Caché local desde el día 1; Stooq como respaldo sin API key |
| Sobre-ingeniería | **Alta** | Ver exclusiones |

---

## Exclusiones explícitas

Nada de esto entra, y cada una se declara como limitación conocida en el README:

- Widgets, dashboards interactivos, notebooks como entregable
- Más de 8 activos
- Costos de transacción
- Restricciones de cartera más allá de `sum(w)=1, w≥0`
- Múltiples niveles de confianza en el backtest principal (99%; el 95% va como tabla
  secundaria solo si sobra tiempo)

Dos exclusiones que merecen su razón, porque no son obvias:

**GARCH-t queda fuera.** Requiere el paquete `arch` y, sobre todo, reajustarse dentro del
walk-forward: ~3,900 reajustes diarios × 8 activos, con sus fallos de convergencia que hay que
manejar uno por uno. Es el sumidero de tiempo clásico de estos proyectos. **EWMA es el primo
no estimado de GARCH**, captura la mayor parte del clustering y cuesta una hora. Si al final
lo quieres, ajústalo mensual en el rebalanceo, nunca diario.

**Cópula-t queda fuera, y no por costo.** El proyecto ya tiene un mecanismo de dependencia de
cola: el salto sistémico (abajo). Un Poisson compartido entre activos *es* una estructura de
dependencia de cola, y sale del propio modelo generador en vez de pegarse por fuera. Añadir
una cópula sería redundante y enturbiaría la tesis.

## Promovido de opcional a plan: el salto sistémico

Un proceso de Poisson **compartido** por todos los activos, además del idiosincrásico de cada
uno. Captura que la correlación se va a 1 en pánico — exactamente la conclusión que el
notebook de performance del curso plantea y nunca modela.

Es la mejora de mayor retorno por hora del proyecto y ahora carga con el rol de estructura de
dependencia de cola. Ubicación natural: al cerrar el día 4, o al inicio del Acto 2.

---

## Resultados esperados

Tres afirmaciones, todas falsables, todas con el número que las respalda:

1. **El VaR bajo GBM subestima el riesgo de cola** porque ignora los saltos — medido en
   excepciones observadas contra esperadas, con veredicto de Kupiec y Christoffersen.
2. **Optimizar por CVaR en vez de por varianza cambia la asignación y reduce la pérdida de
   cola out-of-sample** — medido en el walk-forward.
3. **La elección del modelo de riesgo tiene un precio en capital regulatorio** — medido en
   zonas de semáforo y multiplicador de Basilea, traducido a dólares sobre una posición
   nocional.

Es posible que (2) salga débil o incluso en contra: en algunas ventanas mín-CVaR y
media-varianza casi coinciden. **Si pasa, se reporta tal cual.** Un proyecto que documenta
honestamente que la diferencia fue marginal vale más en una entrevista que uno con un
resultado inflado que no resiste una pregunta.
