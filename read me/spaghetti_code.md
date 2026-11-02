# Evolución del código: del código spaghetti a la arquitectura final

## Evolución cronológica

```mermaid
flowchart LR
    F1["<b>1. Scripts monolíticos</b><br/>24–30/09<br/>Código ejecutado al importar<br/>Sin funciones de integración"]
    F2["<b>2. Extracción de funciones</b><br/>30/09–02/10<br/>euler y runge_kutta_4 como funciones<br/>Primer test de Richardson"]
    F3["<b>3. Interfaz monolítica</b><br/>03–10/10<br/>GUI PySide6 en un solo archivo<br/>De 1382 a 3891 líneas"]
    F4["<b>4. Separación y tests</b><br/>12–25/10<br/>Cálculos movidos a space_physics<br/>test_fisica y Newton"]
    F5["<b>5. Capas</b><br/>27/10<br/>simulation.py sin Qt<br/>Dibujo devuelto a la GUI"]
    F6["<b>6. Paquetes</b><br/>30/10–02/11<br/>Cuatro paquetes y tests/<br/>GUI dividida en módulos"]

    F1 -->|"funciones reutilizables"| F2
    F2 -->|"interacción en tiempo real"| F3
    F3 -->|"física fuera de la GUI"| F4
    F4 -->|"lógica independiente de Qt"| F5
    F5 -->|"modularización"| F6
```

## Arquitectura inicial

Versión del commit `a019b73`, archivo `numerical_engine.py`.

```mermaid
flowchart TD
    A["Parámetros fijos en el script<br/>N, delta, Nv, u0"] --> B["Campo f(u) del oscilador"]
    B --> C["Bucle de Euler a nivel de módulo"]
    C --> D["plt.plot y plt.show"]
    D --> E["Se redefinen F, N, At y u"]
    E --> G["Bucle RK4 repetido<br/>bajo un título de Crank-Nicolson"]
    G --> H["plt.plot y plt.show"]
    I["space_physics.py<br/>euler_equations, jacobian_matrix<br/>kinetic_energy, angular_momentum_sq"]
    I -. "sin conexión con el script" .-> C
```

## Arquitectura final

Las flechas muestran las dependencias principales, no todas las importaciones.

```mermaid
flowchart TD
    subgraph ENT["Entrada"]
        MAIN["main.py y app.py<br/>Arranque"]
        CTRL["controls.py y scenarios.py<br/>I, ω₀, esquema, Δt"]
    end

    subgraph SIM["simulation: une física y esquemas, sin Qt"]
        SESS["simulation.py<br/>SimConfig, Simulation, historial<br/>previsualización e informes"]
        DIV["divergence.py<br/>Detección de divergencia"]
        ERR["error_analysis.py<br/>Derivas, teoría lineal, convergencia"]
    end

    subgraph PHY["space_physics: el modelo físico"]
        MOD["model.py<br/>Euler, invariantes, campo f(u, t)"]
        EXA["exact.py<br/>Solución exacta de Jacobi"]
        EQU["equilibria.py<br/>Equilibrios, autovalores, estabilidad"]
    end

    subgraph NUM["numerical_engine: genérico, no conoce el sólido"]
        NE["numerical_engine.py<br/>Newton, esquemas, ruido de redondeo"]
        INT["integration.py<br/>Problema de Cauchy"]
        STA["stability.py<br/>R(z) obtenida del paso"]
        SCH["schemes.py<br/>Registro SCHEMES"]
        RIC["richardson.py<br/>Orden y error estimado"]
    end

    subgraph SAL["visualization: salida"]
        WRK["workers.py<br/>Cálculo en hilos"]
        WIN["main_window.py<br/>Pestañas y vistas 3D"]
        AUX["geometry.py, verdicts.py, plot_tools.py<br/>Geometría, veredictos, contornos"]
    end

    TST["tests/<br/>test_fisica, test_simulation, test_richardson"]

    MAIN --> WIN
    CTRL -->|"SimConfig"| SESS
    SESS --> DIV
    SESS --> ERR
    SESS -->|"campo f"| MOD
    SESS -->|"referencia"| EXA
    ERR --> EQU
    SESS -->|"esquema y pasos"| INT
    SESS --> RIC
    SCH --> STA
    INT --> NE
    RIC --> INT
    STA --> NE
    EQU --> EXA
    EXA --> MOD
    WRK --> SESS
    SESS -->|"historial y análisis"| WIN
    WIN --> WRK
    WIN --> AUX
    AUX --> EQU
    TST -.-> PHY
    TST -.-> NUM
    TST -.-> SIM
```
