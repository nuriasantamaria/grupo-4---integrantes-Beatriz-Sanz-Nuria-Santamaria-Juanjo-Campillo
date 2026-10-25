# Janibekov Lab

Simulador interactivo del sólido rígido libre (ecuaciones de Euler, efecto Janibekov) con
comparación de esquemas numéricos frente a la solución exacta de Jacobi.

## Arranque

```bash
pip install PySide6 pyqtgraph PyOpenGL numpy
python gui_reactive.py
```

## Estructura

| Archivo | Contenido |
|---|---|
| `gui_reactive.py` | Interfaz (PySide6 + pyqtgraph). Sólo presentación y orquestación. |
| `space_physics.py` | Física: ecuaciones de Euler, energía/momento, equilibrios, **solución exacta** (`EulerExact`), teoría lineal por paso, clasificación del error (`grade`, `noise_floor`), estudio de convergencia. |
| `numerical_engine.py` | Esquemas (Euler, Euler implícito, Crank–Nicolson, RK4), factores de amplificación, Richardson. |
| `test_fisica.py` | Pruebas de física y de `numerical_engine`. |

## Pestañas

- **Dinámica**: ω(t), poloide/polhodia, elipsoide de Poinsot y satélite 3D, con la previsualización de la trayectoria completa.
- **Derivas**: ΔT y ΔL (conservación de energía y momento angular) frente al tiempo.
- **Error vs exacta**: error de módulo r y de fase ψ de cada simulación frente a la solución analítica, con el ruido de redondeo marcado.
- **Plano de fases**: órbitas del flujo reducido, equilibrios y separatrices.
- **Análisis**: Richardson (orden estimado y error), convergencia frente a Δt (pendiente log-log observada vs. nominal) y tabla de conservación.

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
```
