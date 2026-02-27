# 📈 Graph2Formula

Extrae los puntos de una gráfica a partir de una imagen y genera automáticamente su representación analítica (expresión matemática).

```
imagen PNG/JPG  ──►  [(x₁,y₁), (x₂,y₂), ...]  ──►  f(x) = 2·sin(1.57·x + 0.01) + 0.5
```

---

## Módulos

| Archivo | Descripción |
|---|---|
| `graph_extractor.py` | Convierte una imagen de gráfica en una lista de puntos `(x, y)` |
| `analytical_engine.py` | A partir de los puntos, encuentra la expresión analítica que mejor los describe |

---

## Instalación

```bash
pip install opencv-python numpy scipy matplotlib
```

**OCR opcional** (para leer los números de los ejes automáticamente):
```bash
pip install pytesseract pillow
```
Y además instalar Tesseract en el sistema:
- **Windows:** [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux:** `sudo apt install tesseract-ocr`
- **macOS:** `brew install tesseract`

---

## Uso rápido

```python
from graph_extractor   import extraer_puntos
from analytical_engine import analizar

# 1. Extraer puntos de la imagen
puntos = extraer_puntos("mi_grafica.png")

# 2. Obtener la representación analítica
resultado = analizar(puntos)
resultado.mostrar()
resultado.graficar()
```

---

## `graph_extractor.py`

### `extraer_puntos(ruta_imagen, ...)`

**Parámetros**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `ruta_imagen` | `str` | — | Ruta al archivo PNG o JPG |
| `x_min` / `x_max` | `float` | `None` | Rango real del eje X. Si es `None`, se intenta con OCR |
| `y_min` / `y_max` | `float` | `None` | Rango real del eje Y. Si es `None`, se intenta con OCR |
| `suavizado` | `int` | `2` | Nivel de suavizado: `0` ninguno · `1` leve · `2` normal · `3` fuerte |
| `intentar_ocr` | `bool` | `True` | Leer los números de los ejes con Tesseract |
| `guardar_debug_img` | `bool` | `True` | Guardar imagen de diagnóstico en 4 paneles |
| `ruta_debug` | `str` | `"debug_extraccion.png"` | Ruta de la imagen de diagnóstico |

**Retorna:** `list[tuple[float, float]]` — lista de `(x, y)` ordenada por `x`.

### Casos de uso

```python
# Caso A — grid + curva de color + ejes con números → todo automático
puntos = extraer_puntos("grafica_excel.png")

# Caso B — gráfica limpia + curva negra + sin marcas → calibración manual
puntos = extraer_puntos(
    "grafica_papel.png",
    x_min=-10, x_max=10,
    y_min=-5,  y_max=5,
    intentar_ocr=False,
    suavizado=1,
)

# Caso C — exportar a CSV
from graph_extractor import guardar_csv
guardar_csv(puntos, "puntos.csv")
```

### Pipeline interno

```
[1] Carga y normalización   →  redimensiona si > 1400px
[2] Detección de ejes       →  Hough → proyección densidad → fallback margen
[3] Calibración             →  valores manuales > OCR > default [0,1]
[4] Eliminación de grid     →  morfología lineal + inpainting
[5] Aislamiento de curva    →  modo oscuro (umbral adaptativo)
                               modo color (histograma HSV tono dominante)
[6] Extracción subpíxel     →  centroide vertical por columna
                               + reconstrucción cúbica en cruces de ejes
[7] Conversión + suavizado  →  píxeles → coordenadas reales (Savitzky-Golay)
```

### Tipos de imagen soportados

| Tipo | Comportamiento |
|---|---|
| Fondo blanco sin grid | Funciona directamente |
| Fondo con cuadrícula | Se elimina antes de detectar la curva |
| Curva negra / gris | Modo `oscura` — umbralización adaptativa |
| Curva de color (rojo, azul...) | Modo `color` — detección por tono HSV |
| Ejes con números | OCR automático para calibrar el rango real |
| Ejes sin números | Pasar `x_min`, `x_max`, `y_min`, `y_max` manualmente |
| **Curva cruzando los ejes** | Reconstrucción con spline cúbica en el cruce |

### Imagen de diagnóstico

Cuando `guardar_debug_img=True`, se genera `debug_extraccion.png` con 4 paneles:

```
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│  1. Original       │  2. Sin grid       │  3. Máscara        │  4. Puntos         │
│     + ejes         │                    │     de curva       │     extraídos      │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

Úsala para verificar que los ejes se detectaron correctamente antes de continuar.

---

## `analytical_engine.py`

### `analizar(puntos, ...)`

**Parámetros**

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `puntos` | `list[tuple]` | — | Lista de `(x, y)` |
| `umbral_r2` | `float` | `0.97` | R² mínimo para aceptar un tramo |
| `profundidad_max` | `int` | `4` | Máximo de divisiones recursivas (2⁴ = 16 tramos) |
| `verboso` | `bool` | `True` | Imprimir progreso en consola |

**Retorna:** `ResultadoAnalitico` con todos los tramos ajustados.

### Métodos de `ResultadoAnalitico`

```python
resultado.mostrar()          # imprime resumen en consola
resultado.graficar()         # muestra y guarda la gráfica comparativa
resultado.r2_global()        # R² promedio ponderado por longitud de tramo
resultado.tramos             # lista de AjusteParcial con todos los detalles
```

### Catálogo de funciones

| Familia | Tipos incluidos |
|---|---|
| **Polinómica** | Lineal, Cuadrática, Cúbica, Cuártica |
| **Trigonométrica** | Seno, Coseno, Tangente |
| **Exponencial** | Creciente, Decreciente, Gaussiana |
| **Logarítmica** | Logaritmo natural, Logaritmo base 10 |
| **Potencial** | Potencial `xᵇ`, Raíz cuadrada |
| **Racional** | Hiperbólica `1/x`, Sigmoide |

### Algoritmo: divide y vencerás

```
              ┌─────────────────────────────────┐
              │  Analizar forma del tramo        │
              │  (periodicidad, curvatura,       │
              │   monotonía, simetría, FFT...)   │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  Ordenar familias candidatas     │
              │  por probabilidad                │
              └──────────────┬──────────────────┘
                             │
              ┌──────────────▼──────────────────┐
              │  Ajustar funciones con           │
              │  curve_fit (múltiples p₀)        │
              └──────────────┬──────────────────┘
                             │
                      R² ≥ umbral?
                      /           \
                    SÍ             NO  (y prof < máx)
                    │               │
              Aceptar          Dividir en 2
              tramo            y repetir en
                               cada mitad
```

---

## Ejemplo de salida

```
══════════════════════════════════════════════════════
  REPRESENTACIÓN ANALÍTICA
══════════════════════════════════════════════════════

  Tramo 1/2  [-5.0, 0.0]
  Familia   : Trigonométrica
  Tipo      : Seno
  Expresión : f(x) = +1.998·sin(+1.571·x + +0.003) + +0.501
  R²        : 0.9987  ✓ Bueno

  Tramo 2/2  [0.0, 5.0]
  Familia   : Exponencial
  Tipo      : Gaussiana
  Expresión : f(x) = +2.01·exp(-(x-+0.12)²/(2·+1.95²)) + +0.49
  R²        : 0.9941  ✓ Bueno

  R² global ponderado : 0.9964
══════════════════════════════════════════════════════
```

---

## Limitaciones conocidas

- Curvas con **múltiples ramas verticales** (ej. `x = sin(y)`) no se soportan, ya que el extractor mapea una sola fila por columna.
- La detección de ejes puede fallar en imágenes con **bordes o marcos** muy prominentes — en ese caso, pasar los rangos manualmente.
- El OCR requiere que los números en los ejes tengan un **tamaño mínimo** legible (~12px de altura).
- Funciones muy exóticas (ej. `x·sin(1/x)`) pueden requerir aumentar `profundidad_max` o reducir `umbral_r2`.
