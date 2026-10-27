# Janibekov Lab

Simulador interactivo del sólido rígido libre (ecuaciones de Euler, efecto Janibekov) con
comparación de esquemas numéricos frente a la solución exacta de Jacobi.

## Arranque

```bash
pip install PySide6 pyqtgraph PyOpenGL numpy
python gui_reactive.py
```

## Estructura (tres capas)

| Archivo | Capa | Contenido |
|---|---|---|
| `space_physics.py` | Física | Sin Qt ni numérica. Ecuaciones de Euler, invariantes 2T y ‖L‖², cuaternión, **equilibrios con autovalores analíticos** y estabilidad lineal/no lineal, geometría de Poinsot (elipsoides, polhodias, separatriz), cartas del plano de fases, **solución exacta** (`EulerExact`, funciones de Jacobi), teoría lineal por paso, clasificación del error, escenarios predefinidos (`SCENARIOS`), reglas de edición de Ω₀ por ejes. |
| `numerical_engine.py` | Algorítmica | Genérica (no sabe nada del sólido). Esquemas Euler, Euler implícito, Crank–Nicolson y RK4 con su registro `SCHEMES` (orden, R(z), nombre), `cauchy_problem`/`integrate`, Richardson (`richardson_analysis`), regiones de estabilidad (`stability_contour`), utilidades (`n_steps`, `arrow_segments`, `mask_below`). Valida sus entradas. |
| `simulation.py` | Modelo de la GUI (sin Qt) | Une las dos capas: `Simulation` (historial, derivas, error frente a la exacta, divergencia, rebobinado, cambio de esquema al vuelo), `run_preview`, `richardson_report`, `convergence_report`. |
| `gui_reactive.py` | Visualización | PySide6 + pyqtgraph. Solo dibuja y orquesta: sliders, pestañas, 3D, tablas. |
| `test_fisica.py`, `test_simulation.py`, `test_richardson.py` | Pruebas | Física, motor numérico, sesión de simulación y Richardson. |

Todas las funciones de las tres primeras capas llevan *type hints* y pasan `mypy --strict`:
`python -m mypy numerical_engine.py space_physics.py simulation.py --ignore-missing-imports --strict`.

## Equilibrios y estabilidad

En la rotación pura ω = Ω eₖ la Jacobiana tiene autovalores λ = 0 y λ² = Ω²(Iₖ−Iᵢ)(Iⱼ−Iₖ)/(IᵢIⱼ) (`analytic_eigenvalues`, verificado contra la Jacobiana numérica). Imaginarios → centro (ejes extremos); reales ±μ → silla (eje intermedio, Janibekov). El centro es sólo *marginal* en la teoría lineal; la estabilidad **no lineal** (columna «No lineal») sale de T y ‖L‖²: los ejes de mayor y menor inercia son estables y el intermedio inestable.

**Estabilidad de la órbita («Órbita» en las fichas).** Se mide con la órbita exacta (Jacobi): el ángulo máximo que forma ω(t) con su dirección inicial (`max_deviation`). Estable si es menor de 90° (`FLIP_DEG`): girando casi sobre el eje de mayor o menor inercia la desviación es del orden de la perturbación. Inestable («vuelca») si llega a 90° o más: girando casi sobre el eje intermedio llega a ~180° por pequeña que sea la perturbación (teorema del eje intermedio), y lo mismo sobre la separatriz. Marginal: reposo, esfera y sólido simétrico (no hay eje intermedio, ni separatriz).

## Jacobiana exacta en los esquemas implícitos

`space_physics.make_jacobian(I)` da ∂f/∂u (7×7, estado ω + cuaternión) y `simulation` se la pasa a Euler implícito y Crank–Nicolson (`jac=`), de modo que Newton usa la Jacobiana analítica `I − Δ·J` (o `I − Δ/2·J`). Sin `jac` el motor cae a diferencias finitas (`my_jacobian`); ambos caminos dan la misma solución (probado en `test_simulation.py`).

## Preguntas previsibles en la defensa

- **¿Por qué Crank–Nicolson no conserva exactamente T y ‖L‖²?** Conserva exactamente invariantes cuadráticos en problemas *lineales* (|R(iz)| = 1). Aquí el problema es no lineal y la deriva es O(Δt²).
- **¿Por qué Euler explícito gana energía y el implícito la pierde?** |R(iz)| = √(1+z²) > 1 frente a 1/√(1+z²) < 1 en el eje imaginario.
- **¿Por qué el eje intermedio es inestable y no se ve en la linealización de los extremos?** Los extremos tienen λ imaginarios (centro, solo marginal); su estabilidad real se demuestra con T y ‖L‖² (Lyapunov).
- **¿La cuaternión se normaliza?** No durante la integración: su deriva de norma es un indicador más del error del esquema; solo se normaliza al dibujar.
- **¿Qué pasa con Δt grande?** Los explícitos divergen y la simulación se marca como divergida; los implícitos pueden dejar de converger Newton (se registra el error) y el RK4 sale de su región de estabilidad.

## Jacobiana exacta en los esquemas implícitos

`space_physics.make_jacobian(I)` da ∂f/∂u (7×7, estado ω + cuaternión) y `simulation` se la pasa a Euler implícito y Crank–Nicolson (`jac=`): Newton usa `I − Δ·J` (o `I − Δ/2·J`). Sin `jac` el motor cae a diferencias finitas (`my_jacobian`); ambos caminos dan la misma solución (probado en `test_simulation.py`).

## Preguntas previsibles en la defensa

- **¿Por qué Crank–Nicolson no conserva exactamente T y ‖L‖²?** Solo conserva invariantes cuadráticos en problemas lineales (|R(iz)| = 1). Aquí el problema es no lineal y la deriva es O(Δt²).
- **¿Por qué Euler explícito gana energía y el implícito la pierde?** En el eje imaginario |R| = √(1+z²) > 1 (explícito) frente a 1/√(1+z²) < 1 (implícito).
- **¿Cómo se sabe que los ejes extremos son estables si la linealización solo da «centro»?** Un centro es solo marginal; la estabilidad no lineal sale de T y ‖L‖² (Lyapunov).
- **¿Se normaliza el cuaternión?** No durante la integración: su deriva de norma es un indicador más del error del esquema; solo se normaliza al dibujar.
- **¿Qué pasa con Δt grande?** Los explícitos divergen y la simulación se marca como divergida; en los implícitos Newton puede no converger (se registra el error); RK4 sale de su región de estabilidad.

## Cosas que conviene saber al usar la interfaz

- **Cambiar de integrador con la simulación en marcha no reinicia**: el tiempo y el historial siguen y la simulación continúa desde su último estado con el esquema nuevo (también rescata una simulación que había divergido). Cambiar Δt, I o Ω₀ sí reinicia, porque cambia la malla temporal o el problema.
- Richardson y Convergencia se calculan en un hilo (el botón muestra ⏳); si cambias los parámetros mientras calculan, el resultado se descarta.
- **Richardson** (n y 2n pasos) estima el error de ω(t_f) con el orden teórico p del esquema y lo compara con el error real frente a la solución exacta. El **orden observado** se mide en «Convergencia frente a Δt» (pendiente log-log y orden de Richardson con n, 2n y 4n); si las diferencias entre mallas no superan el ruido de redondeo, muestra «—» en vez de números sin sentido.

## Pestañas

- **Dinámica**: ω(t), poloide/polhodia, elipsoide de Poinsot y satélite 3D, con la previsualización de la trayectoria completa.
- **Derivas**: ΔT y ΔL (conservación de energía y momento angular) frente al tiempo.
- **Error vs exacta**: error de módulo r y de fase ψ de cada simulación frente a la solución analítica, con el ruido de redondeo marcado.
- **Plano de fases**: órbitas del flujo reducido, equilibrios y separatrices.
- **Análisis**: tabla de conservación (ΔT y Δ‖L‖), Richardson (error estimado y real), convergencia frente a Δt, tabla de equilibrios (autovalores, tipo lineal y no lineal), efecto de cada esquema sobre el centro y regiones de estabilidad absoluta.

## Cómo leer los errores

- **Módulo** `r`: desviación relativa de la amplitud de la órbita. `r > 0` el esquema inyecta energía de oscilación; `r < 0` la disipa.
- **Fase** `ψ`: adelanto (+) o retraso (−) respecto a la exacta, en grados de ciclo (y su equivalente en segundos).
- Clasificación de cada error:
  - **≈ 0**: por debajo del ruido de redondeo acumulado (≈ 50·ε·√n); ignorable.
  - **ínfimo**: real pero invisible (< 1e-6 en módulo, < 1e-3° en fase).
  - **relevante**: hay que tenerlo en cuenta.
- Teoría en un centro lineal: Euler → módulo crece, fase se retrasa; Euler implícito → módulo decae, fase se retrasa; Crank–Nicolson → módulo se conserva, fase se retrasa; RK4 → errores diminutos (Δt⁵ módulo, Δt⁴ fase).

Las fichas superiores separan tres cosas: **Órbita** (estabilidad física: depende de I y Ω₀), **Esquema** (comportamiento previsto del método con ese Δt) y **Observado** (lo medido).

## Avisos

- *Divergió*: ‖ω‖ supera 1e3 veces lo posible en la órbita exacta; la simulación se congela (las demás siguen).
- *Falló*: `numerical_engine` lanzó una excepción (p. ej. Newton sin converger); se congela igual y el mensaje aparece en la ficha y en la barra de estado.
- *Δt demasiado grande*: Δt·β > 0,25 (menos de ≈ 25 pasos por ciclo).

## Atajos (también en **F1**)

`Espacio` iniciar/pausar · `R` reiniciar · `P` recalcular previsualización · `,` `.` retroceder/avanzar · `Ctrl+1…5` pestañas · `F1` ayuda.

## Pruebas

```bash
python test_fisica.py     # ✔ correcto · ✘ fallo · ⚠ aviso · · omitido
# o:  pytest test_fisica.py
python test_simulation.py
python test_richardson.py
```