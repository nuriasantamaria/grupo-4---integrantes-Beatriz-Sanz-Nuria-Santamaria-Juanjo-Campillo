# Janibekov Lab — Documentación técnica

**Asignatura:** Ampliación de Matemáticas 1 · Máster Universitario en Sistemas Espaciales (MUSE)
**Entrega:** Prueba de Evaluación Intermedia I (PEI 1) · Grupo 4 (B. Sanz, N. Santamaría, J. Campillo)
**Objeto:** simulación y análisis del Teorema del Eje Intermedio (efecto Dzhanibekov) en el sólido rígido libre de pares.

---

## Índice

1. [Arquitectura del sistema](#1-arquitectura-del-sistema)
2. [Análisis de módulos y funciones](#2-análisis-de-módulos-y-funciones)
   - 2.1 [Capa física: `space_physics.py`](#21-capa-física-space_physicspy)
   - 2.2 [Capa matemática y algorítmica: `numerical_engine.py`](#22-capa-matemática-y-algorítmica-numerical_enginepy)
   - 2.3 [Capa de composición: `simulation.py`](#23-capa-de-composición-simulationpy)
   - 2.4 [Capa de visualización dinámica: `gui_reactive.py`](#24-capa-de-visualización-dinámica-gui_reactivepy)
3. [Flujo de ejecución](#3-flujo-de-ejecución)
4. [Cumplimiento de requisitos](#4-cumplimiento-de-requisitos)
5. [Anexo: verificación automática](#5-anexo-verificación-automática)

---

## 1. Arquitectura del sistema

El código sigue la separación en tres capas que pide el enunciado: **Física**, **Matemática/Algorítmica** y **Visualización**. Entre las dos primeras y la interfaz hay un módulo de composición sin dependencias gráficas.

```
┌──────────────────────────────────────────────────────────────────────┐
│  gui_reactive.py          CAPA DE VISUALIZACIÓN (PySide6 + pyqtgraph) │
│  Sliders, pestañas, OpenGL 3D, tablas, hilos de cálculo               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ SimConfig, Simulation, run_preview, …
┌──────────────────────────────▼───────────────────────────────────────┐
│  simulation.py            COMPOSICIÓN (modelo de la GUI, sin Qt)      │
│  Une física y numérica: historial, derivas, divergencia, análisis     │
└───────────────┬──────────────────────────────────────┬───────────────┘
                │ make_rhs, make_jacobian, EulerExact  │ SCHEMES, integrate,
                │ conservation_drift, …                │ richardson_analysis, …
┌───────────────▼──────────────────┐   ┌───────────────▼───────────────┐
│ space_physics.py   CAPA FÍSICA   │   │ numerical_engine.py  CAPA     │
│ Euler, invariantes, Jacobiana,   │   │ MATEMÁTICA: Euler, Euler      │
│ equilibrios, solución exacta     │   │ implícito, Crank–Nicolson,    │
│ (Jacobi), teoría lineal          │   │ RK4, Newton, Richardson, R(z) │
└──────────────────────────────────┘   └───────────────────────────────┘
```

| Módulo | Capa | Depende de | Responsabilidad |
|---|---|---|---|
| `space_physics.py` | Física | solo NumPy | Modelo dinámico del sólido rígido: ecuaciones de Euler, cinemática del cuaternión, invariantes, Jacobiana, equilibrios, solución analítica. |
| `numerical_engine.py` | Matemática | solo NumPy | Integradores de un paso genéricos para $\dot u = f(u,t)$, resolución de Newton, estimación del error (Richardson) y funciones de amplificación $R(z)$. **No sabe que integra un sólido rígido.** |
| `simulation.py` | Composición | las dos anteriores | Sesión de simulación: avance por bloques, historial, derivas, error frente a la exacta, detección de divergencia y análisis pesados. |
| `gui_reactive.py` | Visualización | las tres anteriores + Qt | Entrada interactiva, renderizado 3D y 2D, animación y orquestación de hilos. |

El grafo de dependencias es acíclico y descendente: la física y la numérica no se importan entre sí. `simulation.py` es el único punto que las conecta, inyectando el campo vectorial físico (`make_rhs`) y su Jacobiana (`make_jacobian`) en los integradores genéricos. Por eso cualquier esquema numérico puede probarse sobre otro problema de Cauchy y cualquier modelo físico puede integrarse con otro motor sin tocar código.

**Convenios comunes**

- $I = (I_1, I_2, I_3)$: momentos principales de inercia; $\omega$ en ejes principales del cuerpo.
- Estado completo de 7 componentes: $u = (\omega_1, \omega_2, \omega_3, q_0, q_1, q_2, q_3)$, con el cuaternión de actitud $q = (w, x, y, z)$ escalar primero.
- Una trayectoria es un `ndarray` de forma `(n, 3)` (solo $\omega$) o `(n, 7)` (estado completo).
- Firma universal de un esquema: `scheme(f, u, delta, t) -> u_siguiente`.

---

## 2. Análisis de módulos y funciones

### 2.1 Capa física: `space_physics.py`

El módulo está organizado en diez bloques numerados en su cabecera. Se describen a continuación los que soportan la dinámica, la linealización espectral y la conservación.

#### 2.1.1 Ecuaciones de Euler

Para un sólido libre de pares, en ejes principales:

$$
I_1\dot\omega_1 = (I_2 - I_3)\,\omega_2\omega_3,\qquad
I_2\dot\omega_2 = (I_3 - I_1)\,\omega_3\omega_1,\qquad
I_3\dot\omega_3 = (I_1 - I_2)\,\omega_1\omega_2 .
$$

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `_euler_coefficients(I)` | `I`: matriz diagonal 3×3 | `(c1, c2, c3)` | Calcula $c_i = (I_j - I_k)/I_i$ con $(i,j,k)$ cíclico, de modo que $\dot\omega_i = c_i\,\omega_j\omega_k$. Se calculan una sola vez y los reutilizan la ecuación y su Jacobiana. |
| `euler_equations(omega, I)` | `omega`: vector (3,) o array (3, …); `I`: diag 3×3 | `ndarray` (3,) | Lado derecho de las ecuaciones de Euler. Admite `omega` con dimensiones extra, lo que permite evaluar el flujo sobre una malla completa (lo usa el plano de fases de la GUI). |

#### 2.1.2 Sistema completo: dinámica + cinemática de actitud

Para que la GUI muestre la **orientación** del satélite, la dinámica de $\omega$ se acopla con la cinemática del cuaternión de actitud:

$$
\dot q = \tfrac12\, q \otimes (0, \boldsymbol\omega).
$$

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `rigid_body_rhs(u, t, I_mat)` | `u`: estado (7,); `t`: no se usa (sistema autónomo); `I_mat`: diag 3×3 | `ndarray` (7,) | Campo $f(u,t)$. Las tres primeras componentes son Euler y las cuatro últimas el producto cuaterniónico desarrollado. |
| `make_rhs(I)` | `I`: tres momentos | `Callable[[Array, float], Array]` | Fija `I_mat` con `functools.partial` y devuelve $f(u,t)$ con la firma que esperan los integradores. Valida que haya tres momentos finitos y positivos (`_inertia_matrix` lanza `ValueError` si no). |
| `initial_state(w0)` | `w0`: $\omega(0)$ | `ndarray` (7,) | Estado inicial $(\omega_0, 1, 0, 0, 0)$: actitud identidad. |

El cuaternión **no se renormaliza durante la integración**. Así, la deriva de $\lVert q\rVert$ sirve como indicador adicional del error del esquema; solo se normaliza al dibujar (`quat_to_matrix` en la GUI).

#### 2.1.3 Jacobiana y linealización espectral

La Jacobiana interviene en dos sitios: en el **análisis de estabilidad de los equilibrios** y como **Jacobiana exacta del método de Newton** en los esquemas implícitos.

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `jacobian_matrix(omega, I)` | `omega` (3,), `I` diag 3×3 | `ndarray` (3,3) | $\partial\dot\omega/\partial\omega$ analítica: $J_{ij} = c_i\,\omega_k$ fuera de la diagonal, ceros en la diagonal. |
| `rigid_body_jacobian(u, t, I_mat)` | estado (7,) | `ndarray` (7,7) | Jacobiana completa $\partial f/\partial u$ por bloques. Bloque $\omega\omega$: `jacobian_matrix`. Bloque $q\omega$: $\tfrac12$ de la matriz de $q$. Bloque $qq$: $\tfrac12\,\Omega(\omega)$, la matriz antisimétrica del producto cuaterniónico. Bloque $\omega q$: nulo, porque la dinámica no depende de la actitud. |
| `make_jacobian(I)` | `I` | `Callable` | Pareja exacta de `make_rhs`; `simulation.py` se la pasa a los esquemas implícitos. |

**Autovalores analíticos en los equilibrios.** Las rotaciones puras $\omega = \Omega\,\mathbf e_k$ con $\Omega^2 = 2T/I_k$ son equilibrios. Linealizando en torno a ellas se obtienen los autovalores $\{0, \pm\mu_k\}$ con

$$
\mu_k^2 = \frac{2T}{I_k}\,\frac{(I_k - I_i)(I_j - I_k)}{I_i\, I_j}.
$$

Si $I_k$ es el momento **intermedio**, ambos factores tienen el mismo signo, $\mu_k^2 > 0$, y los autovalores son reales $\pm\mu$: el equilibrio es un **punto de silla**, que es la explicación lineal del efecto Dzhanibekov. Si $I_k$ es extremo, $\mu_k^2 < 0$, los autovalores son imaginarios puros y el equilibrio es un **centro**.

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `principal_equilibria(I, twoT)` | `I`, `twoT` $=2T$ | `ndarray` (3,3) | Fila $k$: $\sqrt{2T/I_k}\,\mathbf e_k$. |
| `analytic_eigenvalues(I, twoT)` | `I`, `twoT` | `ndarray` complejo (3,3) | Fila $k$: $(0, \mu_k, -\mu_k)$, calculado con `np.sqrt(complex(...))` para cubrir los dos signos. El test `test_autovalores_analiticos_coinciden_con_los_numericos` lo contrasta con `np.linalg.eigvals(jacobian_matrix(...))`. |
| `classify_equilibrium(eigvals, tol)` | autovalores | `(tipo, estabilidad_lineal)` | Descarta los autovalores nulos con una tolerancia relativa y clasifica según los signos de la parte real y la presencia de parte imaginaria: punto de silla, nodo o foco estable/inestable, centro o degenerado. |
| `_nonlinear_stability(I, k)` | `I`, eje `k` | `'estable' \| 'inestable' \| 'neutro'` | La linealización de un centro solo da estabilidad **marginal**. La estabilidad no lineal se obtiene de las integrales primeras $T$ y $\lVert L\rVert^2$ (argumento de Lyapunov): los ejes extremos son estables y el intermedio inestable. Si hay empates, devuelve neutro. |
| `equilibrium_analysis(I, twoT)` | `I`, `twoT` | `list[Equilibrium]` | Reúne lo anterior en un `TypedDict` por eje: `omega`, `eigvals`, `type`, `stability` (lineal) y `nonlinear`. |

#### 2.1.4 Invariantes: energía cinética rotacional y momento angular

En ausencia de pares se conservan

$$
2T = \sum_i I_i\,\omega_i^2, \qquad \lVert L\rVert^2 = \sum_i (I_i\,\omega_i)^2 .
$$

Geométricamente, $\omega(t)$ queda en la intersección del **elipsoide de energía** y el **elipsoide de momento angular** (construcción de Poinsot); esa curva intersección es la polhodia.

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `kinetic_energy(omega, I)` | `omega` de forma `(..., 3)`, `I` (3,) | escalar o `ndarray` | $T = \tfrac12\sum I_i\omega_i^2$ vectorizado sobre el último eje, de modo que procesa una trayectoria `(n,3)` entera en una llamada. |
| `angular_momentum_sq(omega, I)` | ídem | escalar o `ndarray` | $\lVert L\rVert^2 = \sum (I_i\omega_i)^2$, igualmente vectorizado. |
| `conserved_quantities(w0, I)` | condición inicial | `(T0, ‖L‖0)` | Valores de referencia del estado inicial. |
| `conservation_drift(W, I, T0, L0)` | trayectoria `W` (n,3), referencias | `(ΔT, Δ‖L‖)`, dos `ndarray` (n,) | **Verificación de la conservación.** Deriva de los invariantes en cada instante respecto al valor inicial. `simulation.py` la evalúa en cada bloque de pasos y la GUI la muestra en la pestaña *Derivas* y en la tabla de conservación de *Análisis*. |
| `omega_bound(I, w0)` | `I`, `w0` | `float` | Cota $\lVert\omega\rVert^2 \le 2T/I_{\min}$, válida en toda la órbita exacta. Sirve de criterio físico de divergencia numérica: si la solución discreta la supera en un factor $10^3$, se da por perdida. |

#### 2.1.5 Solución exacta: funciones elípticas de Jacobi

Para medir el error de los integradores con rigor, el módulo implementa la **solución analítica** de las ecuaciones de Euler sin recurrir a SciPy.

- `_agm(m)`: sucesiones $(a_n, b_n, c_n)$ de la media aritmético-geométrica, con caché LRU.
- `ellip_K(m)`: integral elíptica completa $K(m) = \pi/(2\,a_N)$; devuelve $\infty$ en la separatriz ($m \ge 1 - 10^{-13}$).
- `ellip_F(phi, m)`: integral incompleta $F(\varphi\,|\,m)$ por la transformación descendente de Landen, con corrección de rama para que sea continua en $\varphi$.
- `jacobi(u, m)`: devuelve $(\mathrm{sn}, \mathrm{cn}, \mathrm{dn})$ por la recurrencia AGM inversa. En la separatriz usa los límites $(\tanh u, \operatorname{sech} u, \operatorname{sech} u)$.

**Clase `EulerExact(I, w0)`**

| Miembro | Tipo | Significado |
|---|---|---|
| `kind` | `str` | `'equilibrio'`, `'simetrico'` o `'jacobi'`. |
| `axes` | `[p, q, r]` | `p`: eje extremo cuya componente no cambia de signo; `q`: el otro extremo; `r`: el intermedio. |
| `amp` | `ndarray` (3,) | Amplitudes con signo $(A_p, A_q, A_r)$. |
| `m` | `float` | Parámetro elíptico, obtenido de las dos integrales primeras. |
| `omega_u` | `float` | Frecuencia del argumento $u = u_0 + \sigma\,\omega_u t$. |
| `period` | `float` | $4K(m)/\omega_u$; vale $\infty$ en equilibrio y en la separatriz. |
| `sigma` | `±1` | Sentido de recorrido; se elige comparando la derivada de la solución con $\dot\omega(0)$ de Euler. |

En el constructor se decide qué eje rodea la órbita comparando $\lVert L\rVert^2$ con $2T\,I_{\text{med}}$. Después se obtienen las amplitudes de las relaciones $\omega_p^2 + g_p\,\omega_r^2 = A_p^2$, y de ellas $m$, $\omega_u$ y la fase inicial $u_0 = F(\theta_0\,|\,m)$.

- `omega(t)`: entrada, tiempos de forma arbitraria; salida, `t.shape + (3,)`. Evalúa $\omega_p = A_p\,\mathrm{dn}$, $\omega_q = A_q\,\mathrm{cn}$, $\omega_r = A_r\,\mathrm{sn}$.
- `error_parts(t, W, reset=False)`: compara una trayectoria numérica `W` (n,3) con la exacta y separa el error en dos partes. Devuelve:
  - `r`: error de **módulo** relativo (amplitud de la órbita);
  - `psi`: error de **fase** en grados de ciclo, positivo si el esquema adelanta;
  - `lag`: el mismo error de fase en segundos.

  Mantiene el último ángulo desenrollado (`_theta_last`) para poder llamarla por tandas consecutivas sin saltos de $2\pi$.

#### 2.1.6 Separatriz, teoría lineal del esquema y convergencia

- `separatrix_side(I, w0)`: $(\lVert L\rVert^2 - 2T I_{\text{med}})/(2T I_{\text{med}})$. Positivo: la órbita rodea el eje mayor; negativo: el menor; aproximadamente 0: separatriz.
- `surrounded_axis(I, w0)`: eje extremo que rodea la órbita, o `None` en la separatriz, en sólidos simétricos o en reposo.
- `linear_step_errors(I, w0, amplification, dt) -> LinearErrors | None`: en el centro que rodea la órbita, con frecuencia $\beta = |\operatorname{Im}\mu|$, evalúa $R(i\beta\Delta t)$ del esquema y devuelve el error de módulo por paso `dmod` $= |R| - 1$ y el de fase `dphase` $= \arg R - \beta\Delta t$. Es el enlace teórico entre la física (autovalores) y la numérica (función de amplificación).
- `equilibrium_amplification(...)`: $|R(\lambda\Delta t)|$ en un equilibrio concreto.
- `relevant_eigenvalues(I, w0)`: autovalores no nulos del equilibrio que gobierna la órbita; la GUI los representa como $\lambda\Delta t$ sobre las regiones de estabilidad.
- `noise_floor(n) = 50\,\varepsilon\sqrt n`: umbral de ruido de redondeo acumulado tras `n` pasos.
- `predicted_errors(lin, n)`: errores $(r, \psi)$ previstos por la teoría lineal tras `n` pasos.
- `convergence_study(I, w0, integrate, T, dt, levels) -> Convergence`: integra con $\Delta t_0, \Delta t_0/2, \dots$ y mide el error en $T$ frente a la solución exacta. Descarta los puntos bajo el ruido de redondeo o saturados y ajusta la pendiente log-log (`p`, orden observado), además de las pendientes locales (`local`).

---

### 2.2 Capa matemática y algorítmica: `numerical_engine.py`

Motor **genérico** para el problema de Cauchy $\dot u = f(u,t)$, $u(t_0) = u_0$. Ninguna función conoce el sólido rígido.

#### 2.2.1 Esquemas de un paso

Todos comparten la firma `scheme(f, u, delta, t) -> ndarray`:

| Parámetro | Tipo | Significado |
|---|---|---|
| `f` | `Callable[[Array, float], Array]` | Campo vectorial. |
| `u` | `ndarray` (d,) | Estado en $t_n$. |
| `delta` | `float` | Paso $\Delta t$. |
| `t` | `float` | Instante $t_n$. |
| *retorno* | `ndarray` (d,) | Estado en $t_{n+1} = t_n + \Delta t$. |

| Función | Fórmula | Orden | Coste por paso |
|---|---|---|---|
| `euler` | $u_{n+1} = u_n + \Delta t\, f(u_n, t_n)$ | 1 | 1 evaluación de $f$ |
| `implicit_euler` | $u_{n+1} = u_n + \Delta t\, f(u_{n+1}, t_{n+1})$ | 1 | Newton |
| `crank_nicolson` | $u_{n+1} = u_n + \tfrac{\Delta t}{2}\big[f(u_n,t_n) + f(u_{n+1},t_{n+1})\big]$ | 2 | Newton |
| `runge_kutta_4` | $u_{n+1} = u_n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$ | 4 | 4 evaluaciones de $f$ |

**Resolución de los esquemas implícitos.** Cada paso implícito plantea el sistema no lineal $F(Y) = 0$ y lo resuelve `my_newton`:

- `F(Y)`: para Euler implícito, $Y - u_n - \Delta t\, f(Y, t_{n+1})$; para Crank–Nicolson, $Y - u_n - \tfrac{\Delta t}{2}(f_n + f(Y, t_{n+1}))$.
- **Semilla:** el predictor de Euler explícito $u_n + \Delta t\, f(u_n, t_n)$.
- **Jacobiana de Newton:** si se recibe `jac`, se usa la exacta, $I - \Delta t\,J$ o $I - \tfrac{\Delta t}{2}J$. Si no, `my_jacobian` la aproxima por diferencias hacia delante con paso relativo $h_j = 10^{-7}\max(1, |Y_j|)$.
- **Parada:** $\lVert\Delta Y\rVert \le \text{tol}\cdot\max(1, \lVert Y\rVert)$, con `tol = 1e-12` y `max_iter = 20`.
- **Fallos:** lanza `ArithmeticError` si aparecen NaN/inf y `RuntimeError` si no converge. Las excepciones se propagan; `simulation.py` las captura y congela la simulación afectada con el mensaje de error, sin falsear resultados.

#### 2.2.2 Registro de esquemas y funciones de amplificación

Cada esquema se registra con sus propiedades teóricas en `SCHEMES: dict[str, Scheme]`, de modo que la GUI y los análisis leen de un único sitio el orden, la función $R(z)$ y el nombre.

| Clave | `amplification` $R(z)$ | `order` | `module_order` | `phase_order` | `implicit` |
|---|---|---|---|---|---|
| `euler` | $1 + z$ | 1 | 1 | 2 | No |
| `be` | $1/(1 - z)$ | 1 | 1 | 2 | Sí |
| `cn` | $(1 + z/2)/(1 - z/2)$ | 2 | `None` ($\lvert R\rvert = 1$) | 2 | Sí |
| `rk4` | $\sum_{k=0}^{4} z^k/k!$ | 4 | 5 | 4 | No |

`module_order` y `phase_order` son los órdenes globales en $\Delta t$ del error de $|R|$ y de $\arg R$ en el eje imaginario, $z = i\beta\Delta t$, que es donde están los autovalores de un centro. Explican el comportamiento observado: Euler explícito **inyecta** energía ($|R| = \sqrt{1+\beta^2\Delta t^2} > 1$), Euler implícito la **disipa** ($|R| < 1$), Crank–Nicolson conserva el módulo en el problema lineal y RK4 tiene errores de módulo $O(\Delta t^5)$ y de fase $O(\Delta t^4)$.

#### 2.2.3 Integración: gestión del paso de tiempo

El motor usa **paso fijo** $\Delta t$; no hay control adaptativo del paso. Es una decisión deliberada: con paso constante el orden del error se mide directamente y las propiedades espectrales $R(\lambda\Delta t)$ se pueden comparar.

| Función | Entrada | Salida | Descripción |
|---|---|---|---|
| `cauchy_problem(f, u0, t, scheme, stop=None)` | malla `t` (N+1,), condición inicial | `ndarray` (N+1, d) | Bucle de integración sobre una malla arbitraria: $\Delta t_n = t_{n+1} - t_n$. Si `stop(u)` devuelve `True`, rellena el resto con NaN y corta. |
| `integrate(scheme, f, u0, dt, n, t0, limit, components)` | `dt`, `n` pasos | `ndarray` (n+1, d) | Construye la malla uniforme y aplica un criterio de corte: se considera divergencia si $\sum\lvert u_{\text{components}}\rvert$ deja de ser finita o alcanza `limit`. Valida `dt > 0` finito y `n ≥ 0` entero. |
| `first_divergence(Y, bound, components)` | trayectoria, cota | `int \| None` | Primera fila no finita o con norma mayor que `bound`. |
| `final_state(scheme, f, u0, T, N, t0)` | horizonte `T`, `N` pasos | `ndarray` (d,) | Estado en $t_0 + T$. |
| `n_steps(T, dt)` | | `int` | Número de pasos (al menos 1) que cubren `T`. |

#### 2.2.4 Estimación del error: extrapolación de Richardson

Al ser el paso fijo, el error se estima **a posteriori** comparando mallas anidadas. Si $u_h$ tiene error $C h^p$:

$$
u(T) - u_{h/2} \approx \frac{u_{h/2} - u_h}{2^p - 1}.
$$

| Función | Salida | Descripción |
|---|---|---|
| `scheme_order(scheme, f, u0, T, N, components)` | `float` | **Orden observado** $p \approx \log_2\big(\lVert u_N - u_{2N}\rVert / \lVert u_{2N} - u_{4N}\rVert\big)$. Devuelve NaN si las diferencias caen bajo el umbral de redondeo `_rounding_floor` $= 50\,\varepsilon\sqrt{n}\max(1, \lVert u\rVert)$, para no presentar un orden falso. |
| `richardson_error(..., p)` | `ndarray` (N+1, d) | Error estimado de la solución fina en todos los nodos de la malla gruesa. |
| `richardson_extrapolation(..., p)` | `ndarray` (d,) | Solución extrapolada $u_{2N} + (u_{2N} - u_N)/(2^p - 1)$, de orden $p+1$. |
| `richardson_analysis(scheme, f, u0, T, N, p, components, reference, t0)` | `RichardsonReport` | Análisis completo en una sola pasada: `n`, `p`, `err` (error estimado), `u_extr`, `err_real` (error real si se proporciona `reference(T)`, que en este proyecto es la solución exacta de Jacobi) y `noise` (señala si la estimación está dominada por el redondeo). |

`_check_richardson` valida que $T > 0$, $N \ge 1$ entero y $p > 0$, y lanza `ValueError` con un mensaje explícito en caso contrario.

#### 2.2.5 Estabilidad numérica

La **estabilidad absoluta** se evalúa sobre el problema test $\dot y = \lambda y$: el esquema es estable si $|R(\lambda\Delta t)| \le 1$.

- `stability_grid(amplification, re, im, n) -> (x, y, A)`: evalúa $A_{ij} = |R(x_i + i\,y_j)|$ en una malla $n\times n$ vectorizada. Donde $R$ no está definida (polo de Crank–Nicolson o de Euler implícito) asigna $+\infty$.
- La GUI extrae de esta malla la frontera $|R| = 1$ (`level_curve`, *marching squares*) y superpone los puntos $\lambda\Delta t$ con los autovalores **reales** del sólido (`relevant_eigenvalues`). Así se ve directamente si el paso elegido deja el espectro del problema dentro o fuera de la región de estabilidad.
- En tiempo de ejecución, la estabilidad se vigila con dos mecanismos: el corte por `limit` dentro de `integrate` y la cota física `omega_bound` aplicada por `first_divergence` en `simulation.py`.

---

### 2.3 Capa de composición: `simulation.py`

Es el modelo que consume la interfaz y no depende de Qt. Su función es conectar la física con el motor numérico.

| Elemento | Descripción |
|---|---|
| `STEPS`, `step_for(method, I)` | Obtiene el paso del registro `SCHEMES`. Si el esquema es implícito, le inyecta `jac=make_jacobian(I)` con `partial`, de modo que Newton usa la Jacobiana analítica 7×7. |
| `SimConfig` (dataclass) | `name`, `color`, `enabled`, `method` (clave de `SCHEMES`), `I`, `w0`, `dt`. Es la unidad de configuración que produce la GUI. |
| `integrate(method, y0, dt, I, n, t0)` | `ne.integrate` aplicado al sólido, con `limit = DIVERGENCE_LIMIT = 1e6`. |
| `linear_errors(cfg)` | Teoría lineal del esquema en la órbita configurada. |

**Clase `Simulation(cfg)`**

- **Estado interno:** búferes preasignados `_Y` (n,7), `_dT`, `_dL`, `_er`, `_ep`, `_el`, con capacidad inicial `CHUNK = 8192` que se duplica (`_grow`) cuando hace falta. Así se evita copiar arrays en cada paso.
- `reset()`: vuelve a $t = 0$, recalcula $T_0$, $\lVert L\rVert_0$, la cota `w_bound` y crea el objeto `EulerExact`.
- `advance(n)`: integra `n` pasos desde el último estado, con estos pasos internos:
  1. Captura cualquier excepción del motor (p. ej. Newton sin converger) y marca `diverged = True` y `error = "<tipo>: <mensaje>"`.
  2. Recorta la trayectoria en la primera fila que supere $10^3\cdot$`w_bound` (`first_divergence`).
  3. Calcula y guarda $\Delta T$, $\Delta\lVert L\rVert$ (`conservation_drift`) y los errores de módulo y fase (`EulerExact.error_parts`).
- `advance_to(t_target)`: avanza hasta el instante pedido, con un máximo de `MAX_STEPS_PER_CALL = 20000` pasos por llamada para no bloquear la interfaz.
- `switch_method(method)`: cambia el integrador **sin reiniciar**; el historial se conserva y la integración continúa desde el último estado válido.
- **Rebobinado:** `view_n` y las propiedades `vY`, `vdT`, `vt`, `vy`, … exponen el historial hasta un instante pasado sin borrarlo.

**Análisis pesados** (los ejecutan los hilos de la GUI):

- `run_preview(cfg, T_total, should_cancel, yield_s)`: trayectoria completa hasta el horizonte, en bloques de 2000 pasos. Es cancelable y entre bloques cede el GIL. Devuelve un `Preview` (`TypedDict`) o `None` si se cancela.
- `richardson_report(cfg, T)`: `richardson_analysis` con la solución exacta como referencia.
- `convergence_report(cfg, T, levels)`: `convergence_study` más el orden de Richardson.
- `run_each(cfgs, job, should_cancel)`: aplica un análisis a cada simulación activa. Si una falla, se guarda su excepción y las demás continúan.

#### 2.3.1 Funciones de `space_physics.py` que usa `simulation.py`

`simulation.py` importa de la capa física exactamente estos nombres:

```python
from space_physics import (ArrayLike, Convergence, EulerExact, LinearErrors, conservation_drift, conserved_quantities,
                           convergence_study, initial_state, make_jacobian, make_rhs, omega_bound,
                           linear_step_errors)
```

**Uso directo**

| Nombre de `space_physics` | Dónde lo usa `simulation.py` | Para qué |
|---|---|---|
| `make_rhs(I)` | `integrate`, `richardson_report`, `convergence_report` | Campo $f(u,t)$ del sólido que se pasa a los integradores. |
| `make_jacobian(I)` | `step_for` | Jacobiana exacta 7×7 para el Newton de Euler implícito y Crank–Nicolson. |
| `initial_state(w0)` | `Simulation.reset`, `richardson_report`, `convergence_report` | Estado inicial $(\omega_0, 1, 0, 0, 0)$. |
| `conserved_quantities(w0, I)` | `Simulation.reset` | Valores de referencia $T_0$ y $\lVert L\rVert_0$. |
| `conservation_drift(W, I, T0, L0)` | `Simulation.advance` | Derivas $\Delta T$ y $\Delta\lVert L\rVert$ de cada bloque de pasos. |
| `omega_bound(I, w0)` | `Simulation.reset` | Cota física de $\lVert\omega\rVert$ que, multiplicada por `DIVERGENCE_FACTOR`, define la divergencia. |
| `EulerExact(I, w0)` | `Simulation.reset`, `richardson_report` | Solución exacta de referencia. |
| `EulerExact.error_parts(t, W)` | `Simulation.advance` | Errores de módulo y fase frente a la exacta. |
| `EulerExact.omega(t)` | `richardson_report` | Estado exacto en $T$ para calcular `err_real`. |
| `linear_step_errors(I, w0, R, dt)` | `linear_errors(cfg)` | Teoría lineal del esquema en el centro de la órbita. |
| `convergence_study(I, w0, integrate, T, dt, levels)` | `convergence_report` | Error en $T$ frente a $\Delta t$ y orden observado. |
| `ArrayLike`, `LinearErrors`, `Convergence` | Anotaciones de tipo | Solo tipos; no se ejecuta código. |

**Uso indirecto** (funciones que no se importan, pero se ejecutan a través de las anteriores)

| Llamada directa | Arrastra a |
|---|---|
| `make_rhs` | `_inertia_matrix`, `rigid_body_rhs` → `euler_equations` → `_euler_coefficients` |
| `make_jacobian` | `_inertia_matrix`, `rigid_body_jacobian` → `jacobian_matrix` → `_euler_coefficients` |
| `conserved_quantities`, `conservation_drift` | `kinetic_energy`, `angular_momentum_sq` |
| `omega_bound` | `kinetic_energy` |
| `EulerExact` | `euler_equations`, `sorted_axes`, `ellip_K`, `ellip_F`, `jacobi`, `_agm`, `SEP_M` |
| `linear_step_errors` | `_center` → `surrounded_axis` (→ `is_degenerate`, `kinetic_energy`, `separatrix_side`, `sorted_axes`, `SEPARATRIX_TOL`) y `equilibrium_analysis` (→ `analytic_eigenvalues`, `principal_equilibria`, `classify_equilibrium`, `_nonlinear_stability`) |
| `convergence_study` | `EulerExact`, `noise_floor` (→ `EPS`) |

**No las usa `simulation.py`** (ni directa ni indirectamente): `relevant_eigenvalues`, `equilibrium_amplification` (con su tipo `EquilibriumAmplification`) y `predicted_errors`. Solo las usan `gui_reactive.py` y los tests.

---

### 2.4 Capa de visualización dinámica: `gui_reactive.py`

Interfaz construida con **PySide6** (Qt 6), **pyqtgraph** para las gráficas 2D y **pyqtgraph.opengl** para las escenas 3D. El módulo no integra ecuaciones: dibuja y orquesta. Al principio contiene funciones de apoyo geométrico que son puramente de visualización y que las capas inferiores no usan.

#### 2.4.1 Organización de la ventana

`MainWindow` (subclase de `QMainWindow`) se divide en:

- **Panel izquierdo** (`_build_header`, `_build_controls`): botones Iniciar/Pausa y Reiniciar, velocidad de reproducción, tiempo de parada, selector de escenarios didácticos (`SCENARIOS`), problema físico compartido (`PhysicsSliders`) y tres pestañas A/B/C de configuración por simulación (`SimConfigPanel`).
- **Banner superior** (`_build_banner`): una ficha por simulación con tres diagnósticos: **Órbita** (estabilidad física), **Esquema** (lo que predice la teoría lineal con ese $\Delta t$) y **Observado** (el error medido).
- **Pestañas de vistas** (`_build_view_tabs`, accesibles con Ctrl+1…5): *Dinámica*, *Derivas*, *Error vs exacta*, *Plano de fases* y *Análisis*.
- **Línea de tiempo** (`_build_timeline`): rebobinado y avance paso a paso.

#### 2.4.2 Renderizado del espacio de fases 3D y superficies de Poinsot

Vista `phase_view` (`GLViewWidget`) de la pestaña *Dinámica*.

| Función / método | Qué dibuja | Cómo |
|---|---|---|
| `poinsot_semiaxes(I, w0) -> (semi_E, semi_L)` | Semiejes de los elipsoides | Energía: $\sqrt{2T/I_i}$. Momento: $\lVert L\rVert/I_i$. |
| `_build_phase_items()` | Crea los objetos 3D persistentes | Ejes $\omega_1, \omega_2, \omega_3$; `mesh_E` y `mesh_L` (esferas translúcidas con *shader* `balloon` y mezcla aditiva); puntos de equilibrio; y, para cada simulación, la órbita exacta, la estela, la previsualización y la cabeza. |
| `_update_static_phase()` | Reconstruye la geometría al cambiar $I$ u $\Omega_0$ | Escala los vértices de una esfera unidad por los semiejes de Poinsot. Añade las curvas de `separatrix_curves` y la familia de polhodias de `polhode_family`. Ajusta los ejes y la cámara. |
| `separatrix_curves(I, twoT)` | Las dos separatrices | Parametrización cerrada que pasa por los equilibrios del eje intermedio. |
| `polhode_family(I, twoT, n_levels)` | Polhodias alrededor de los ejes extremos | Niveles de $\lVert L\rVert^2$ entre $2T I_{\min}$ y $2T I_{\max}$ intersecados con el elipsoide de energía. |
| `_update_exact_orbits()` | Órbita exacta | `exact_orbit(EulerExact)` parametrizada por el ángulo de amplitud (resuelve bien la zona cercana a la separatriz) y dibujada como **tubo** (`tube`), porque el grosor de las líneas GL depende del controlador gráfico. |
| `_update_equilibria()` | Puntos $\pm\Omega_k\mathbf e_k$ | Coloreados por estabilidad (verde estable, azul marginal, rojo inestable). Rellena además las tablas de equilibrios y de efecto del esquema. |
| `_refresh_views()` (rama *Dinámica*) | Estela y cabeza | Últimos `trail_spin / dt` puntos de `sim.vY[:, :3]`. La estela tiene la misma duración en segundos para todas las simulaciones aunque sus $\Delta t$ difieran. |

Las casillas *Energía*, *Momento*, *Separatriz*, *Polhodias* y *Exacta* muestran u ocultan cada capa (`_update_static_visibility`). Si alguna simulación tiene otra física y no está sobre las superficies dibujadas, `_update_surface_notice` lo advierte e incluye un enlace para cambiar de referencia.

La trayectoria numérica debería permanecer **sobre la intersección** de los dos elipsoides. Cualquier separación visible es deriva de los invariantes, de modo que esta vista es una comprobación cualitativa directa de la conservación.

#### 2.4.3 Actualización en tiempo real de la orientación del satélite

Vista `sat_view` (`GLViewWidget`) de la pestaña *Dinámica*, en el marco **inercial**.

1. **Geometría del cuerpo** (`_rebuild_satellites`, `_make_body`): `body_half_sizes(I)` obtiene los semiejes de una caja homogénea con esos momentos de inercia,
   $a_1 \propto \sqrt{I_2 + I_3 - I_1}$, etc., normalizados. Así la forma visible corresponde físicamente a la inercia elegida. Formas disponibles: Satélite (cuerpo y paneles), Paralelepípedo, Elipsoide, Vaca, Caza Estelar y OVNI. Las mallas solo se reconstruyen si cambia algo del cuerpo (clave `_sat_key`), porque es la operación más costosa.
2. **Actitud en cada fotograma** (`_update_satellites`): para cada simulación se toma el estado mostrado `sim.vy` y se llama a
   `attitude_vectors(q, I, w, L0_norm, w0_norm) -> (R, L_dir, w_dir)`:
   - `R = quat_to_matrix(q)`: matriz cuerpo→inercial, con el cuaternión normalizado solo aquí;
   - `L_dir = R·(I ω)/‖L‖₀` y `w_dir = R·ω/‖ω₀‖`: vectores en el marco inercial.

   La matriz `R` se empaqueta en una `Transform3D` 4×4 junto con el desplazamiento de la simulación y se aplica a todas las mallas del cuerpo. Se redibujan los ejes principales $\mathbf e_1, \mathbf e_2, \mathbf e_3$ (columnas de `R`), el vector **L** (amarillo) y **ω** (blanco).

En el sistema libre, **L** es constante en el marco inercial: debe verse **fijo** mientras el cuerpo vuelca alrededor de él. Si se desplaza, el integrador no conserva el momento angular. En la configuración *Efecto Janibekov*, los vuelcos periódicos de 180° del cuerpo se ven directamente.

#### 2.4.4 Controles interactivos y modificación de variables en tiempo de ejecución

| Clase | Controles | Rango | Señal |
|---|---|---|---|
| `FloatSlider` | `QSlider` + `QDoubleSpinBox` sincronizados | configurable | `valueChanged(float)` |
| `PhysicsSliders` | $I_1, I_2, I_3$; $\Omega_0$ por eje principal más una o dos perturbaciones, o por componentes | $I \in [0.1, 6.0]$ | `changed()` |
| `SimConfigPanel` | Activa/inactiva, esquema (combo con `SCHEMES`), $\Delta t$, vínculo con la física compartida o física propia | $\Delta t \in [0.001, 0.25]$ | `changed()` |
| `MainWindow` | Velocidad, *Parar en*, horizonte de previsualización, escenario, referencia | Velocidad $\in [0.2, 30]$ | — |

`FloatSlider` sincroniza el deslizador (entero) y la caja numérica (real) bloqueando temporalmente las señales (`blockSignals`), lo que evita bucles de realimentación. `PhysicsSliders` etiqueta cada momento como *mín/medio/máx* y avisa, sin impedirlo, cuando se viola la desigualdad triangular $I_k \le I_i + I_j$.

**Cadena de propagación de un cambio:**

```
FloatSlider.valueChanged ─► PhysicsSliders / SimConfigPanel.changed
        ─► MainWindow._schedule_apply()           (QTimer de un disparo, 120 ms: antirrebote)
        ─► MainWindow.apply_params()
              ├─ si solo cambió el esquema ─► _switch_schemes(): Simulation.switch_method()
              │                               (el tiempo y el historial continúan)
              └─ en otro caso ─► nuevas Simulation(cfg) en t = 0
                                  _update_static_phase, _update_exact_orbits, _update_equilibria,
                                  _update_phase_plane, _rebuild_satellites, request_preview, …
```

El antirrebote de 120 ms evita reconstruir la escena con cada píxel de arrastre. Cambiar $\Delta t$, $I$ u $\Omega_0$ **reinicia** la simulación porque cambia la malla temporal o el problema. Cambiar solo el integrador **no la reinicia**, lo que permite comparar esquemas sobre la misma historia.

La velocidad se aplica de otra forma: no reinicia nada y solo cambia cuánto tiempo simulado avanza cada fotograma (§3).

#### 2.4.5 Resto de vistas

- **Derivas** (`_build_drift_tab`): $\Delta T(t)$, $\Delta\lVert L\rVert(t)$ y $\omega(t)$ de la referencia, con escala logarítmica opcional y cursor vertical enlazado entre gráficas (`LinkedCursor`).
- **Error vs exacta** (`_build_error_tab`, `_refresh_error_tab`): componente de $\omega$ numérica frente a la exacta, error de módulo $r(t)$ y de fase $\psi(t)$ (en grados de ciclo o segundos), con el suelo de ruido de redondeo marcado.
- **Plano de fases** (`_update_phase_plane`): carta $(\omega_i, \omega_j)$ sobre un hemisferio de $\omega_k$, con $\omega_k$ despejado de la energía. Dibuja el dominio admisible, el campo de direcciones (`reduced_flow_field` + `arrow_segments`), las curvas de nivel analíticas (elipses alrededor de un centro, hipérbolas alrededor de una silla) y el equilibrio con su clasificación.
- **Análisis** (`_build_analysis_tab`): tabla de conservación; Richardson (error estimado frente a real); convergencia frente a $\Delta t$; tabla de equilibrios (autovalores, tipo lineal y no lineal); efecto de cada esquema en el centro; y **regiones de estabilidad absoluta** (`_update_stability`, con contornos en caché mediante `stability_layer`) con los puntos $\lambda\Delta t$ de cada simulación.

#### 2.4.6 Concurrencia

Los cálculos costosos no bloquean la interfaz:

- `PreviewWorker(QThread)`: ejecuta `run_preview` para cada simulación activa. Cada petición lleva un **token** creciente; `_on_preview_done` descarta resultados obsoletos.
- `AnalysisWorker(QThread)`: ejecuta Richardson o el estudio de convergencia mediante `run_each`. Al lanzar un análisis del mismo tipo se cancela el anterior, y `_on_analysis_done` comprueba que el emisor sigue siendo el vigente.
- `closeEvent` cancela y espera a todos los hilos antes de cerrar.

---

## 3. Flujo de ejecución

El punto de entrada es la interfaz:

```bash
pip install PySide6 pyqtgraph PyOpenGL numpy
python gui_reactive.py
```

1. **Importación de módulos.** Al importar `gui_reactive.py` se cargan `numerical_engine` (se construye `SCHEMES`), `space_physics` y `simulation` (se construye `STEPS`). La GUI deriva de `SCHEMES` sus etiquetas, colores y órdenes (`METHOD_LABEL`, `SCHEME_ORDER`).
2. **`main()`**
   1. Configura pyqtgraph (antialiasing, colores de fondo y primer plano).
   2. Crea la `QApplication` con estilo *Fusion*, elige tipografías disponibles (`pick_font`) y aplica la hoja de estilo (`build_stylesheet`).
   3. Instancia `MainWindow()`.
3. **`MainWindow.__init__`**
   1. Inicializa el estado: lista de simulaciones vacía, reloj `clock = 0`, `running = False`, registro de hilos y cachés.
   2. `_build_ui()`: construye el panel de control, el banner, las cinco pestañas y la línea de tiempo.
   3. `_build_phase_items()` y `_build_plot_items()`: crea una sola vez todos los objetos gráficos (mallas, líneas, curvas); después solo se actualizan sus datos.
   4. Crea el temporizador de antirrebote `_param_timer` (120 ms).
   5. `apply_preset(DEFAULT_SCENARIO)` carga el escenario *«Comparar integradores (mismo Δt)»*: $I = (1, 2, 3)$, $\Omega_0 = (0.01, 2.0, 0.01)$ (rotación casi pura sobre el eje intermedio), $\Delta t = 0.02$, con Euler, Crank–Nicolson y RK4 en A, B y C.
   6. `apply_params()` crea tres objetos `Simulation` en $t = 0$; calcula los elipsoides, la separatriz, las polhodias, las órbitas exactas, los equilibrios y el plano de fases; construye los satélites; y lanza la previsualización en segundo plano.
   7. Arranca el temporizador de animación (`FRAME_MS = 30` ms, unos 33 fotogramas por segundo) y registra los atajos de teclado (Espacio, R, P, `,`, `.`, Ctrl+1…5, F1).
4. **Bucle de eventos** (`app.exec()`). La simulación empieza en pausa. Al pulsar **Iniciar** (o Espacio), `toggle_run()` pone `running = True`.
5. **Cada fotograma** (`_tick`):
   1. Calcula el tiempo simulado del fotograma: $\text{velocidad} \times 0.030$ s.
   2. `_advance_live(step)` avanza el reloj global (respetando *Parar en*) y llama a `sim.advance_to(clock)` en cada simulación activa. Ahí `Simulation.advance` → `ne.integrate` → `scheme(f, u, Δt, t)` → `rigid_body_rhs`. Después se calculan las derivas y los errores frente a la exacta y se comprueba la divergencia.
   3. Notifica en la barra de estado las simulaciones que divergen (quedan congeladas y las demás continúan) y pausa automáticamente si todas han divergido o se alcanza el tiempo límite.
   4. `_refresh_views()` actualiza **solo la pestaña visible**: en *Dinámica*, estelas 3D, actitud de los satélites y marcadores de la exacta. Cada 4 fotogramas actualiza las fichas y la barra de estado.
6. **Interacción.** Cualquier cambio de controles sigue la cadena de §2.4.4. La línea de tiempo permite rebobinar (`_set_view_time` fija `view_n` en cada simulación) y reproducir el historial sin recalcularlo.
7. **Cierre.** `closeEvent` cancela los hilos activos y espera a que terminen.

---

## 4. Cumplimiento de requisitos

**Rigor dinámico**

- Se integran las ecuaciones de Euler exactas del sólido libre, acopladas con la cinemática cuaterniónica de la actitud. No hay ángulos de Euler, que tendrían singularidades.
- La linealización usa la **Jacobiana analítica**, y los autovalores en los equilibrios se obtienen en **forma cerrada** y se verifican contra los numéricos. El efecto Dzhanibekov se justifica en tres niveles complementarios:
  1. **Lineal:** el eje intermedio es un punto de silla.
  2. **No lineal:** los ejes extremos son estables por las integrales primeras $T$ y $\lVert L\rVert^2$.
  3. **Global:** con la órbita exacta, el ángulo máximo de $\omega(t)$ respecto a $\omega_0$ supera 90° en el eje intermedio.
- La conservación de $2T$ y $\lVert L\rVert^2$ se mide en cada paso, y se representa a la vez cuantitativamente (derivas, tabla) y geométricamente (trayectoria sobre la intersección de los elipsoides de Poinsot; $\mathbf L$ fijo en el marco inercial).
- Se implementa la **solución exacta de Jacobi** (AGM propia, sin SciPy) como referencia. Así los errores se miden de forma absoluta y se separan en módulo y fase, no solo de forma relativa entre esquemas.

**Métodos numéricos**

- Cuatro esquemas de un paso: dos explícitos (Euler, RK4) y dos implícitos (Euler inverso, Crank–Nicolson). Los implícitos se resuelven con Newton propio, con Jacobiana exacta o por diferencias finitas, y con un tratamiento explícito de la no convergencia.
- Estimación del error por extrapolación de Richardson, orden observado con tres mallas y estudio de convergencia log-log frente a la solución exacta. En los tres casos se detecta el umbral de redondeo para no presentar resultados sin significado.
- Análisis de estabilidad absoluta con las funciones de amplificación $R(z)$ y los autovalores físicos $\lambda\Delta t$ superpuestos, más una teoría lineal por paso ($|R| - 1$, $\arg R - \beta\Delta t$) que **predice** el comportamiento que luego se **observa**.

**Modularidad**

- Tres capas con dependencias en un único sentido. `numerical_engine.py` es independiente del problema físico y `space_physics.py` es independiente del método numérico; `simulation.py` es el único punto de composición y la GUI no contiene lógica de integración.
- Un registro único de esquemas (`SCHEMES`): añadir un integrador consiste en escribir su función de paso y su $R(z)$ y registrarlo; la GUI lo incorpora automáticamente.
- Las funciones de las capas física, numérica y de composición llevan anotaciones de tipo completas (`numpy.typing`, `TypedDict`, `NamedTuple`) y validan sus entradas con excepciones explícitas (`ValueError`, `ArithmeticError`, `RuntimeError`).

---

## 5. Anexo: verificación automática

| Archivo | Pruebas | Qué cubre |
|---|---|---|
| `test_fisica.py` | 41 | Ecuaciones de Euler, solución exacta frente a integración fina, conservación, separatriz, autovalores analíticos frente a numéricos, clasificación de equilibrios, teoría lineal, órdenes de convergencia, funciones de amplificación, Newton, regiones de estabilidad, validación de entradas, inestabilidad del eje intermedio frente a los extremos. |
| `test_simulation.py` | 12 | Coincidencia con la exacta, avance por bloques equivalente a un avance único, historial y reinicio, divergencia, registro de excepciones, cambio de esquema al vuelo, previsualización cancelable, Jacobiana exacta frente a diferencias finitas en Newton, rechazo de inercias no positivas. |
| `test_richardson.py` | 3 | Orden de RK4, estimación de Richardson y mejora por extrapolación. |

Ejecución (en Windows conviene forzar UTF-8 en la consola, porque la salida usa los símbolos ✔/✘):

```bash
set PYTHONIOENCODING=utf-8      # PowerShell: $env:PYTHONIOENCODING="utf-8"
python test_fisica.py
python test_simulation.py
python test_richardson.py
```

En la revisión de este documento (27/10/2026) las **56 pruebas pasan**.
