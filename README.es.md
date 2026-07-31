# Motor de asignación y riesgo con saltos

*[English version](README.md) · [Guía técnica](docs/guia.html)*

## ¿Cuánto capital te cuesta el modelo de riesgo equivocado?

**Menos. Ese es el problema.**

Auditoría de **10 especificaciones de VaR/ES × 4 carteras** sobre 8 clases de activo y 3,923
días out-of-sample (2011-2026), con validación de Kupiec, Christoffersen, Acerbi-Székely y el
semáforo de capital de Basilea.

---

## Resultado

| | |
|---|---|
| **El VaR gaussiano produce 2.06× las excepciones esperadas** | Reprueba Kupiec, cobertura condicional y el backtest de ES en las 4 carteras |
| **El VaR con saltos produce 0.97×** | La mejor calibración de frecuencia de las diez |
| **39 de 40 combinaciones reprueban independencia** | Tras una excepción la probabilidad se multiplica por 16 |
| **Ningún modelo acierta frecuencia y momento a la vez** | El mejor en frecuencia es casi el peor en agrupamiento, y viceversa |
| **El modelo que reprueba todo cuesta 18.4% MENOS capital** | El castigo del semáforo no compensa lo que se ahorra subestimando |
| **Optimizar CVaR no redujo la cola realizada** | Asigna 10 pp distinto que mínima varianza, con +0.1% de CVaR |

**Recomendación.** Adoptar **`mc_merton`** como especificación de riesgo. De las diez es la
única que acierta a la vez la frecuencia de las pérdidas extremas (0.96×, la mejor calibrada) y
su magnitud (pasa el backtest de Expected Shortfall en las 4 carteras) — y la magnitud es
justamente lo que fija el capital bajo Basilea III.

Su defecto es real y hay que decirlo: **no modela agrupamiento de volatilidad**, así que sus
excepciones se apelmazan en crisis. Ningún modelo del conjunto resuelve los tres ejes; el que
mejor acierta el momento (`ewma`, `fhs`) falla la magnitud. **La extensión evidente —y no
construida— es acoplar las marginales de Merton a una escala condicional tipo EWMA.**

Y no confiar en el semáforo de Basilea como único criterio de aprobación: sobre estos datos
**premia económicamente a los modelos que reprueba**. La conclusión práctica no es afinar el
modelo para pasar el test, sino que el test cuantitativo no basta.

---

## Método

**Datos.** 8 ETF (`SPY QQQ IWM EFA EEM TLT GLD DBC`), cierres ajustados, 2007-01-03 a
2026-07-30. 4,924 días sin huecos. Descarga única cacheada en `data/prices.csv`.

**Modelo.** Difusión con saltos de Merton sobre retornos logarítmicos diarios:

```
X = m + σ·Z + Σ_{i=1}^{N} Y_i        N ~ Poisson(λ),  Y_i ~ N(μ_J, σ_J²)
```

Calibrado por método de momentos: los cumulantes en forma cerrada se igualan a la varianza,
skew y kurtosis muestrales. La calibración reproduce los momentos empíricos a `rtol=1e-6`.

**Salto sistémico.** Un único proceso de Poisson compartido por los ocho activos, con tamaños
de salto correlacionados por la matriz empírica. Como λ es la misma en todos, compartir la
cuenta no altera ninguna marginal — se gana dependencia de cola sin recalibrar nada.

**Walk-forward.** Ventana rodante de 1,000 días. Los pesos se reoptimizan al cambiar de mes
(188 rebalanceos); el VaR se recalcula todos los días sobre los pesos vigentes. Reoptimizar a
diario invalidaría la medición: el VaR estaría midiendo una cartera que nunca existió un día
completo.

**Diez especificaciones.** Cinco incondicionales sin volatilidad variable (`historico`,
`normal`, `mc_gbm`, `mc_merton`, `mc_merton_idio`), tres con colas no gaussianas pero varianza
plana (`t_student`, `cornish_fisher`, `evt`) y dos con volatilidad condicional (`ewma`, `fhs`).
Todas se registran bajo una firma única `(retornos, pesos, nivel, rng) → (VaR, ES)`, de modo que
el bucle de backtest no conoce ningún modelo por nombre.

**Validación.** Kupiec (frecuencia), Christoffersen (independencia y cobertura condicional),
Acerbi-Székely Test 2 (Expected Shortfall) y el semáforo de capital de Basilea. Las pruebas de
cobertura se verificaron contra series sintéticas de respuesta conocida **antes** de aplicarlas
a los datos reales: Kupiec rechaza 9/100 bajo H₀ cierta y 100/100 con la tasa inflada;
Christoffersen rechaza 100/100 series apelmazadas con tasa incondicional correcta, donde Kupiec
solo rechaza 24/100.

El de Acerbi-Székely recibió el mismo tratamiento y **no lo pasó a la primera**: su distribución
nula se construía remuestreando los cocientes de cola observados, lo que la contaminaba con la
alternativa — con el ES un 50% subestimado rechazaba 4 de 100. El nulo se reconstruyó desde el
modelo (el exceso sobre el VaR como exponencial de media ES − VaR, la elección de máxima
entropía consistente con ese par) y ahora la potencia crece con la severidad: 1/60 bajo H₀,
12/60 con datos t(6) y 36/60 con t(3). **La corrección cambió veredictos**, y las tablas de
este informe son las posteriores.

---

## Acto 1 — el diagnóstico

*(Cinco especificaciones incondicionales. El Acto 2 amplía a diez y matiza la conclusión.)*

### 1. Los saltos corrigen la frecuencia de excepciones

![excepciones](figures/1_excepciones_acumuladas.png)

| Modelo | Excepciones | Razón obs/esp | Kupiec |
|---|---|---|---|
| `normal` | 81 | **2.06×** | REPRUEBA |
| `mc_gbm` | 81 | **2.06×** | REPRUEBA |
| `historico` | 52 | 1.33× | pasa |
| `mc_merton` | 38 | **0.97×** | pasa |

*(cartera de mínima varianza; el patrón se repite en las cuatro — ver
[figura 3](figures/3_razon_excepciones.png))*

`normal` y `mc_gbm` coinciden casi exactamente, como deben: son el mismo supuesto por dos vías,
una cerrada y otra por simulación. Que converjan es una verificación cruzada que no se diseñó.

### 2. Las colas idiosincrásicas se diversifican; las sistémicas no

Con saltos **independientes** entre activos, el cociente ES/VaR de la cartera es 1.159 —
indistinguible del gaussiano (1.162). Los choques independientes se promedian entre ocho
activos y la cola vuelve a ser normal. Solo el salto **sistémico** sobrevive a la agregación
(1.460).

Esto se refleja en el backtest: `mc_merton_idio` pasa Kupiec en las carteras concentradas y lo
reprueba en `risk_parity` e `igual_peso`, las dos más diversificadas.

**Consecuencia de diseño:** modelar colas gordas activo por activo no sirve para el riesgo de
una cartera. Lo que importa es si saltan juntos.

### 3. Acertar la frecuencia no es acertar el momento

![var vs realizado](figures/2_var_vs_realizado.png)

**Las cinco especificaciones de este acto reprueban la prueba de independencia**, incluida la
que pasa Kupiec con 0.97×. Los diagnósticos:

```
π_01 = 0.0082      π_11 = 0.1351      →  16.4× tras una excepción

Excepciones por año:   2012: 0    2020: 10-16
                       2014: 0    2018:  5-12
                       2017: 0    2022:  4-5
```

Las cinco peores pérdidas del periodo son cuatro días de marzo de 2020 y el 8 de agosto de 2011.

No es un defecto de implementación: **ninguna de las cinco tiene volatilidad variable en el
tiempo** — todas usan una ventana plana de 1,000 días, así que estructuralmente no pueden
capturar agrupamiento. El veredicto es correcto y marca el límite del enfoque incondicional.

Ese límite es la pregunta que abre el Acto 2: **¿condicionar por volatilidad lo arregla?**

---

## Acto 2 — auditoría de las diez especificaciones

Promedio sobre las 4 carteras. `pasa` cuenta pruebas superadas de 16 (4 carteras × Kupiec,
independencia, cobertura condicional y Acerbi-Székely).

| modelo | razón | persistencia | ES (de 4) | multiplicador | capital (k$/10M) | pasa (de 16) |
|---|---|---|---|---|---|---|
| `mc_merton` | **0.96** | 20.4 | **4** | 3.11 | 551 | **8** |
| `t_student` | 1.08 | 12.1 | 3 | 3.11 | 550 | 7 |
| `historico` | 1.20 | 12.5 | 2 | 3.12 | 518 | 6 |
| `cornish_fisher` | 0.68 | 29.6 | **4** | 3.07 | 692 | 6 |
| `evt` | 1.08 | 14.4 | 2 | 3.12 | 531 | 6 |
| `fhs` | 1.22 | **7.5** | 2 | 3.10 | 503 | 5 |
| `mc_merton_idio` | 1.37 | 9.5 | 0 | 3.17 | 485 | 2 |
| `ewma` | 2.04 | **3.6** | 0 | 3.34 | 447 | 1 |
| `normal` | 1.78 | 7.0 | 0 | 3.24 | 466 | **0** |
| `mc_gbm` | 1.79 | 7.0 | 0 | 3.24 | 466 | **0** |

### Tres ejes, y ningún modelo acierta los tres

![frontera](figures/5_frontera.png)

La pregunta del Acto 2 era si la volatilidad condicional arregla la falla de independencia del
Acto 1. La respuesta es parcial, y al validarla aparece un tercer eje que separa a los
candidatos de forma tajante:

| | Frecuencia | Momento | Magnitud (ES) | Mecanismo |
|---|---|---|---|---|
| `ewma` | **peor** (2.04×) | **mejor** (π 3.6) | 0/4 | vol condicional, colas gaussianas |
| `fhs` | buena (1.22×) | 2ª mejor (π 7.5) | 2/4 | escala condicional, forma empírica |
| `mc_merton` | **mejor** (0.96×) | casi el peor (π 20.4) | **4/4** | colas gordas, vol plana |

De 40 combinaciones, **una sola pasa independencia**: `igual_peso/ewma`, con p = 0.146. EWMA
reduce la persistencia a 3.6 pero falla la frecuencia porque sigue suponiendo normalidad: sabe
*cuándo* sube el riesgo y no *cuánta* cola tiene.

FHS parecía la síntesis —escala condicional de EWMA más forma empírica de la cola— y acierta la
frecuencia y el momento. Pero **falla la magnitud**: reprueba el backtest de ES en 2 de 4
carteras (p = 0.006 y 0.021). El mecanismo es coherente con su construcción: reescala residuos
históricos por la volatilidad vigente, así que un salto que golpea durante un tramo de calma se
mide con residuos estandarizados en calma y la severidad sale corta.

Merton es el inverso exacto: acierta cuánto y no cuándo. Como el capital regulatorio bajo
Basilea III se fija con el ES, **la magnitud pesa más que el momento**, y por eso es la
recomendación pese a su peor persistencia.

### El incentivo invertido

![incentivo](figures/6_incentivo.png)

Capital = multiplicador × VaR medio, sobre $10M de nocional, cartera de mínima varianza:

```
mc_gbm         $389,062   +0.0%   ← reprueba Kupiec, CC y ES
normal         $389,877   +0.2%   ← reprueba las cuatro pruebas
ewma           $393,288   +1.1%
fhs            $427,270   +9.8%
mc_merton      $460,524  +18.4%   ← el mejor calibrado
cornish_fisher $530,557  +36.4%
```

**El modelo que reprueba todo es el más barato.** El mecanismo: el multiplicador castiga como
máximo +33% (3.00 → 4.00), mientras que subestimar el VaR ahorra ~20% de forma directa. Y el
castigo efectivo queda muy por debajo del tope: con el doble de la tasa de excepciones, un
modelo promedia 5.2 excepciones por ventana de 250 días — apenas la primera banda amarilla. **El
semáforo tiene poca potencia en ventanas de 250 días.**

Esto no es un artefacto del cálculo: es la crítica documentada al marco de backtesting de
Basilea, y una de las razones por las que Basilea III (FRTB) trasladó la métrica de capital de
VaR a ES y conserva requisitos cualitativos de aprobación además del test cuantitativo.

*La hipótesis original de este proyecto era que reprobar el backtest encarecería el capital.
Resultó incorrecta en el signo. Se reporta como salió.*

---

## El resultado negativo

**Optimizar CVaR no redujo el riesgo de cola realizado.**

| | ret_anual | vol_anual | sharpe | VaR99 | CVaR99 | peor_día | max_dd |
|---|---|---|---|---|---|---|---|
| `min_cvar` | 7.07% | 9.03% | 0.783 | 1.52% | 2.08% | **4.55%** | 23.75% |
| `min_var` | 6.97% | 8.72% | **0.799** | 1.52% | 2.08% | 5.02% | **22.67%** |
| `risk_parity` | 6.79% | 9.47% | 0.717 | 1.63% | 2.36% | 5.55% | 23.89% |
| `igual_peso` | 7.67% | 12.61% | 0.608 | 2.20% | 3.24% | 7.99% | 25.04% |

CVaR realizado: **+0.1%**. Máximo drawdown: **+4.8% peor**. Sharpe: ligeramente menor. La única
métrica donde gana es el peor día individual (−9.3%).

**No es que los optimizadores coincidan.** Verifiqué que las carteras son genuinamente
distintas: la diferencia de pesos es de 10 puntos porcentuales, contra una dispersión entre
semillas de simulación de 1.8 pp — señal real, 5.5× por encima del ruido. El optimizador de
CVaR hace algo distinto; simplemente no paga.

La lectura más plausible es error de estimación. CVaR al 99% se ajusta al 1% extremo de una
distribución que a su vez se estimó con 1,000 días. Al bajar a β=0.95 —donde la cola tiene
cinco veces más datos— la diferencia de pesos cae de 10 pp a 3 pp. La discrepancia vive
justamente donde los datos son más escasos.

*Esto se reporta como salió. Una hipótesis previa mía —que la construcción con salto sistémico
volvía la distribución conjunta elíptica y por eso las soluciones coincidían— resultó
incorrecta al contrastarla, y quedó descartada.*

---

## Limitaciones declaradas

- **λ no está identificada** por los momentos 2 a 4: son 5 parámetros contra 4 ecuaciones.
  Se fija en 0.05/día (12.6 saltos/año) como decisión de modelado declarada, con sensibilidad
  reportada en `merton.sensibilidad_lambda`. Cerrarla exigiría el sexto cumulante muestral, que
  es ruido.
- **Sin costos de transacción.** Con rebalanceo mensual y carteras estables el efecto sería
  modesto, pero no está medido.
- **GARCH no está incluido.** Reajustarlo dentro del walk-forward son ~3,900 estimaciones
  diarias por activo con sus fallos de convergencia. EWMA es su primo no estimado y captura la
  mayor parte del agrupamiento; FHS lo usa como filtro. Es la extensión natural.
- **Sin dependencia de cola asimétrica.** El salto sistémico acopla a los ocho activos con la
  misma intensidad. Cargas de salto distintas por activo darían una estructura más rica —
  y es el candidato más probable a que la optimización de CVaR sí pague.
- **Solo posiciones largas**, `sum(w)=1, w≥0`.
- **Un solo nivel de confianza** (99%) en el backtest principal.
- **Calibración por momentos, no por máxima verosimilitud.** El MLE de Merton tiene una
  verosimilitud no acotada conocida; se prefirió el estimador robusto.

---

## Reproducir

```bash
pip install -r requirements.txt
python tests/test_core.py      # 21 checks de validación, ~4 min
python -m src.backtest         # walk-forward, 10 modelos × 4 carteras, ~8 min
python -m src.basel            # semáforo de capital y backtest de ES
python -m src.plots            # figuras
python -m src.diario --sembrar # corrida diaria con estado (modo sombra)
```

`tests/test_core.py` contiene el criterio de validación de cada fase. Los cuatro que más
importan:

1. **La calibración reproduce los cumulantes en forma cerrada** (`rtol=1e-6`), sin ruido de
   simulación — aísla la calibración del simulador.
2. **El `α` del programa lineal converge al VaR empírico**: 0.015208 contra 0.015208. El LP
   nunca ve un percentil; si hubiera un error de signo o escala, saltaría aquí.
3. **Bajo escenarios gaussianos centrados, mín-CVaR converge a mínima varianza** (máx|Δw| =
   0.0022) — la equivalencia teórica que valida la formulación de Rockafellar-Uryasev.
4. **Las pruebas de cobertura se validan contra series sintéticas** antes de usarse.

## Estructura

```
src/data.py       ingesta y caché
src/merton.py     calibración, cumulantes, simulación, escenarios conjuntos
src/optimize.py   mín-CVaR (LP), Markowitz (QP), risk parity
src/risk.py       registro de los 10 modelos VaR/ES con firma única
src/backtest.py   walk-forward, Kupiec, Christoffersen
src/basel.py      semáforo de capital y Acerbi-Székely
src/diario.py     corrida diaria con estado persistente
src/plots.py      figuras
```

## Hipótesis descartadas

Tres conjeturas propias que los datos refutaron, documentadas porque probar y descartar es
parte del trabajo:

| Hipótesis | Qué dijeron los datos |
|---|---|
| El salto sistémico vuelve elíptica la distribución conjunta, y por eso mín-CVaR coincide con mínima varianza | Falso: el caso de mezcla pura dio la **mayor** diferencia de pesos, no la menor |
| Mín-CVaR reduce la cola realizada frente a media-varianza | Falso: +0.1% de CVaR y 4.8% peor drawdown, pese a asignar 10 pp distinto |
| Reprobar el backtest encarece el capital regulatorio | **Invertido**: el modelo que reprueba las cuatro pruebas cuesta 18.4% menos |
