"""Texto HTML de la ayuda"""

from __future__ import annotations


HELP_HTML = """
<h2>Atajos</h2>
<table cellspacing="6">
<tr><td><b>Espacio</b></td><td>iniciar / pausar</td></tr>
<tr><td><b>R</b></td><td>reiniciar (t = 0)</td></tr>
<tr><td><b>P</b></td><td>recalcular la previsualización</td></tr>
<tr><td><b>,</b> &nbsp; <b>.</b></td><td>retroceder / avanzar un paso de vista</td></tr>
<tr><td><b>Ctrl+1 … Ctrl+5</b></td><td>cambiar de pestaña</td></tr>
<tr><td><b>F1</b></td><td>esta ayuda</td></tr>
</table>

<h2>Las tres líneas de cada ficha de estabilidad</h2>
<ul>
<li><b>Órbita</b>: propiedad <i>física</i>; sólo depende de I y de Ω₀. Mide cuánto se aleja el eje de giro
de su dirección inicial a lo largo de la órbita exacta.
<span style="color:{ok}">estable</span>: se desvía poco (menos de 90°), del orden de la perturbación inicial:
es lo que ocurre girando casi sobre el eje de mayor o menor inercia.
<span style="color:{bad}">inestable</span>: <i>vuelca</i>; el eje llega a más de 90° (casi 180° para perturbaciones
pequeñas) de donde empezó, por pequeña que sea la perturbación: es el teorema del eje intermedio, y también
ocurre sobre la separatriz. <span style="color:{mar}">marginal</span>: reposo, esfera o sólido simétrico
(no hay eje intermedio).</li>
<li><b>Esquema</b>: lo que la teoría lineal prevé para el método numérico elegido y Δt en el centro
que rodea la órbita (módulo y fase por paso, extrapolado al horizonte de la previsualización).
No depende del eje: es el comportamiento del <i>método</i>.</li>
<li><b>Observado</b>: lo medido en la simulación frente a la solución exacta, en el instante mostrado.</li>
</ul>

<h2>Error de módulo y de fase</h2>
<p>La órbita exacta se recorre con una amplitud y una velocidad angular de fase. El
<b>error de módulo</b> r es la desviación relativa de la amplitud (r &gt; 0: el esquema inyecta energía de
oscilación; r &lt; 0: la disipa). El <b>error de fase</b> ψ es cuánto se adelanta (+) o se retrasa (−) la
solución numérica sobre la exacta, en grados de ciclo (y su equivalente en segundos).</p>
<p>Cada error se clasifica comparándolo con el ruido de redondeo de la máquina y con lo
visible en una gráfica:</p>
<ul>
<li><span style="color:{g0}"><b>≈ 0</b></span>: no se distingue del redondeo de la máquina; se puede ignorar.</li>
<li><span style="color:{g1}"><b>ínfimo</b></span>: real, pero por debajo de 10<sup>−6</sup> (módulo) o 10<sup>−3</sup>° (fase);
invisible en cualquier gráfica.</li>
<li><span style="color:{g2}"><b>relevante</b></span>: hay que tenerlo en cuenta.</li>
</ul>
<p>Teoría (centro lineal): Euler explícito → módulo crece, fase se retrasa; Euler implícito → módulo
decae, fase se retrasa; Crank–Nicolson → módulo se conserva (|R| = 1), fase se retrasa; RK4 → error
minúsculo en ambos (Δt⁵ en módulo, Δt⁴ en fase).</p>

<h2>Otros avisos</h2>
<ul>
<li><b>Divergió</b>: ‖ω‖ supera 10³ veces lo posible en la órbita exacta; esa simulación se congela y las
demás continúan. La simulación se pausa sólo cuando divergen todas las activas.</li>
<li><b>Falló</b>: <code>numerical_engine</code> lanzó una excepción (p. ej. Newton sin converger); se congela igual
y el mensaje del error aparece en la ficha y en la barra de estado.</li>
<li><b>Δt demasiado grande</b>: Δt·β &gt; 0.25, es decir, menos de ≈ 25 pasos por ciclo de la oscilación.</li>
<li><b>Convergencia</b>: pendiente log-log del error frente a Δt; debe coincidir con el orden teórico del
esquema mientras el error esté por encima del ruido de redondeo.</li>
</ul>
"""
