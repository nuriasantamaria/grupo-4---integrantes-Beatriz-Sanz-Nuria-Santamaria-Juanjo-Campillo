"""Capa de visualización (PySide6 + pyqtgraph)

Sin Qt (se pueden importar y probar sin interfaz):
  palette.py             colores
  schemes.py             etiquetas, colores y órdenes de los esquemas; región de estabilidad en caché
  geometry.py            forma y actitud del cuerpo, Poinsot, polhodias, separatriz, plano de fases
  initial_conditions.py  edición de ω₀ por ejes y comparación de problemas
  plot_tools.py          escalas, máscaras, flechas y contorno de la región de estabilidad
  verdicts.py            gravedad de los errores y estabilidad de la órbita
  cells.py               celdas de tabla
  scenarios.py           escenarios predefinidos y configuración inicial
  help_text.py           texto de la ayuda

Con Qt:
  theme.py               fuentes, hoja de estilos, estilos HTML/CSS
  widgets.py             tablas, tarjetas, layouts, cursor enlazado
  controls.py            sliders y paneles de parámetros
  workers.py             hilos de previsualización y análisis
  skins/                 cuerpos 3D (satélite, vaca, ovni, caza TIE) y primitivas de malla
  window/                ventana principal
  app.py                 main(): crea la aplicación Qt y abre la ventana
"""


def main() -> None:
    """Arranca la interfaz; Qt solo se carga al llamarla"""
    from visualization.app import main as run
    run()
