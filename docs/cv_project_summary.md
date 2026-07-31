# Motor de Riesgo Cuantitativo — VaR/ES con saltos sistémicos

*Proyecto personal · Python, cálculo estocástico, optimización convexa, validación estadística*

## Para el CV — dos líneas

- Construí un motor de riesgo de mercado para 8 ETF que compara **10 modelos VaR/ES** y **4
  estrategias de asignación** mediante walk-forward de **3,923 días fuera de muestra**
  (2011–2026).
- Implementé **jump-diffusion de Merton** con acoplamiento sistémico, **optimización CVaR** por
  programación lineal y validación con **Kupiec, Christoffersen y Acerbi-Székely**; identifiqué
  que el VaR gaussiano alcanzó hasta **2.06× las excepciones esperadas**.

## Qué demuestra

| Competencia | Dónde |
|---|---|
| **Cálculo estocástico** | Difusión con saltos de Merton calibrada por método de momentos sobre cumulantes en forma cerrada; proceso de Poisson compartido para dependencia de cola |
| **Optimización convexa** | Mín-CVaR como programa lineal (Rockafellar-Uryasev), Markowitz como QP, risk parity por la formulación convexa de Spinu-Maillard |
| **Validación estadística** | Cuatro pruebas de hipótesis, cada una verificada contra series sintéticas de respuesta conocida *antes* de aplicarse a datos reales |
| **Rigor metodológico** | Walk-forward sin fuga de información, intervalos bootstrap por bloques, sensibilidad acotada a los parámetros elegidos por juicio |
| **Ingeniería** | Registro de modelos con firma única, manifiesto de procedencia con SHA-256, CI en dos versiones de Python, suite de 25 comprobaciones |

## Los tres resultados

1. **El VaR gaussiano subestima la cola.** 2.06× las excepciones esperadas y reprueba las cuatro
   pruebas en las cuatro carteras. El de saltos da 0.97× y pasa la validación de magnitud.

2. **Las colas idiosincrásicas se diversifican; las sistémicas no.** Con saltos independientes el
   cociente ES/VaR de la cartera es 1.159, indistinguible del gaussiano (1.162). Modelar colas
   gordas activo por activo no sirve para el riesgo de una cartera.

3. **El modelo peor calibrado sale más barato.** Bajo el semáforo VaR histórico, el modelo que
   reprueba las cuatro pruebas exige 18.4% menos capital que el mejor calibrado, porque el
   castigo del multiplicador no compensa lo que se ahorra subestimando.

## Lo que también demuestra: honestidad metodológica

Tres hipótesis propias quedaron refutadas por los datos y están documentadas como tales — la
optimización CVaR **no** redujo la cola realizada, y el signo del hallazgo de capital resultó
inverso al esperado. Una prueba estadística del propio proyecto (Acerbi-Székely) falló su
validación interna, se corrigió, y la corrección **cambió veredictos y la recomendación final**.

## Alcance declarado

Es un estudio comparativo de modelos, **no un cálculo de capital regulatorio**. El semáforo
reproducido es el histórico basado en VaR (Basilea II/2.5), usado como vara de comparación; el
marco vigente (FRTB) calcula el IMA con Expected Shortfall al 97.5% e incorpora atribución de
P&L, factores no modelizables y recargos supervisores, todo fuera de alcance.

## Reproducir

```bash
make venv && make test && make backtest && make report
```

Datos cacheados en el repositorio: la suite corre sin red. Manifiesto con hashes SHA-256,
fecha de corte, parámetros y versiones en `data/manifest.json`.
