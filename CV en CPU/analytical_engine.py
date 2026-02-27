"""
====================================================
  MOTOR DE REPRESENTACIÓN ANALÍTICA
====================================================
  INPUT : Lista de puntos (x, y) extraídos de gráfica
  OUTPUT: Expresión(es) analítica(s) que aproximan
          la curva, con su dominio y precisión

  Estrategia:
    1. Analizar la FORMA del tramo (curvatura,
       periodicidad, monotonía, simetría...)
    2. Clasificar en una FAMILIA de funciones
    3. Ajustar parámetros con curve_fit
    4. Si R² < umbral → dividir en 2 y repetir
       (hasta MAX_PROFUNDIDAD divisiones)

  Dependencias:
    pip install numpy scipy matplotlib

  Uso rápido:
    from analytical_engine import analizar
    resultado = analizar(puntos)
    resultado.mostrar()
====================================================
"""

import numpy as np
import warnings
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from dataclasses import dataclass, field
from typing import Callable, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────
R2_ACEPTABLE   = 0.97    # R² mínimo para considerar un ajuste "bueno"
R2_TOLERABLE   = 0.92    # R² para aceptar si no hay mejor opción
MAX_PROFUNDIDAD = 4       # Máximo de divisiones recursivas (2^4 = 16 tramos)
MIN_PUNTOS      = 10      # Mínimo de puntos para intentar ajuste


# ─────────────────────────────────────────────────
# CATÁLOGO DE FAMILIAS Y FUNCIONES
# ─────────────────────────────────────────────────
#
#  Cada entrada es:
#    nombre    : etiqueta legible
#    familia   : grupo al que pertenece
#    funcion   : callable f(x, *params)
#    p0        : valores iniciales de parámetros
#    latex     : plantilla LaTeX con slots {a},{b}...
#    bounds    : límites para curve_fit (opcional)

CATALOGO: list[dict] = [

    # ── POLINÓMICAS ────────────────────────────────
    {
        "nombre" : "Lineal",
        "familia": "Polinómica",
        "funcion": lambda x, a, b: a*x + b,
        "p0"     : [1.0, 0.0],
        "latex"  : "{a}·x + {b}",
    },
    {
        "nombre" : "Cuadrática",
        "familia": "Polinómica",
        "funcion": lambda x, a, b, c: a*x**2 + b*x + c,
        "p0"     : [1.0, 0.0, 0.0],
        "latex"  : "{a}·x² + {b}·x + {c}",
    },
    {
        "nombre" : "Cúbica",
        "familia": "Polinómica",
        "funcion": lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d,
        "p0"     : [1.0, 0.0, 0.0, 0.0],
        "latex"  : "{a}·x³ + {b}·x² + {c}·x + {d}",
    },
    {
        "nombre" : "Cuártica",
        "familia": "Polinómica",
        "funcion": lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e,
        "p0"     : [1.0, 0.0, 0.0, 0.0, 0.0],
        "latex"  : "{a}·x⁴ + {b}·x³ + {c}·x² + {d}·x + {e}",
    },

    # ── TRIGONOMÉTRICAS ────────────────────────────
    {
        "nombre" : "Seno",
        "familia": "Trigonométrica",
        "funcion": lambda x, a, b, c, d: a * np.sin(b*x + c) + d,
        "p0"     : [1.0, 1.0, 0.0, 0.0],
        "latex"  : "{a}·sin({b}·x + {c}) + {d}",
    },
    {
        "nombre" : "Coseno",
        "familia": "Trigonométrica",
        "funcion": lambda x, a, b, c, d: a * np.cos(b*x + c) + d,
        "p0"     : [1.0, 1.0, 0.0, 0.0],
        "latex"  : "{a}·cos({b}·x + {c}) + {d}",
    },
    {
        "nombre" : "Tangente",
        "familia": "Trigonométrica",
        "funcion": lambda x, a, b, c, d: a * np.tan(b*x + c) + d,
        "p0"     : [1.0, 0.5, 0.0, 0.0],
        "latex"  : "{a}·tan({b}·x + {c}) + {d}",
    },

    # ── EXPONENCIALES ──────────────────────────────
    {
        "nombre" : "Exponencial creciente",
        "familia": "Exponencial",
        "funcion": lambda x, a, b, c: a * np.exp(b*x) + c,
        "p0"     : [1.0, 0.5, 0.0],
        "latex"  : "{a}·eˢ({b}·x) + {c}",
    },
    {
        "nombre" : "Exponencial decreciente",
        "familia": "Exponencial",
        "funcion": lambda x, a, b, c: a * np.exp(-abs(b)*x) + c,
        "p0"     : [1.0, 0.5, 0.0],
        "latex"  : "{a}·e⁻({b}·x) + {c}",
    },
    {
        "nombre" : "Gaussiana",
        "familia": "Exponencial",
        "funcion": lambda x, a, mu, sigma, d: a * np.exp(-((x - mu)**2) / (2*sigma**2)) + d,
        "p0"     : [1.0, 0.0, 1.0, 0.0],
        "latex"  : "{a}·exp(-(x-{mu})²/(2·{sigma}²)) + {d}",
    },

    # ── LOGARÍTMICAS ───────────────────────────────
    {
        "nombre" : "Logaritmo natural",
        "familia": "Logarítmica",
        "funcion": lambda x, a, b, c: a * np.log(np.abs(b*x) + 1e-9) + c,
        "p0"     : [1.0, 1.0, 0.0],
        "latex"  : "{a}·ln({b}·x) + {c}",
    },
    {
        "nombre" : "Logaritmo base 10",
        "familia": "Logarítmica",
        "funcion": lambda x, a, b, c: a * np.log10(np.abs(b*x) + 1e-9) + c,
        "p0"     : [1.0, 1.0, 0.0],
        "latex"  : "{a}·log₁₀({b}·x) + {c}",
    },

    # ── POTENCIAL ──────────────────────────────────
    {
        "nombre" : "Potencial",
        "familia": "Potencial",
        "funcion": lambda x, a, b, c: a * np.power(np.abs(x) + 1e-9, b) + c,
        "p0"     : [1.0, 2.0, 0.0],
        "latex"  : "{a}·|x|^{b} + {c}",
    },
    {
        "nombre" : "Raíz cuadrada",
        "familia": "Potencial",
        "funcion": lambda x, a, b, c: a * np.sqrt(np.abs(b*x) + 1e-9) + c,
        "p0"     : [1.0, 1.0, 0.0],
        "latex"  : "{a}·√(|{b}·x|) + {c}",
    },

    # ── RACIONAL ───────────────────────────────────
    {
        "nombre" : "Hiperbólica 1/x",
        "familia": "Racional",
        "funcion": lambda x, a, b, c: a / (x + b + 1e-9) + c,
        "p0"     : [1.0, 0.01, 0.0],
        "latex"  : "{a} / (x + {b}) + {c}",
    },
    {
        "nombre" : "Sigmoide",
        "familia": "Racional",
        "funcion": lambda x, a, b, c, d: a / (1 + np.exp(-b*(x - c))) + d,
        "p0"     : [1.0, 1.0, 0.0, 0.0],
        "latex"  : "{a} / (1 + e^(-{b}·(x-{c}))) + {d}",
    },
]

# Índice rápido por familia
FAMILIAS = {}
for entrada in CATALOGO:
    FAMILIAS.setdefault(entrada["familia"], []).append(entrada)


# ─────────────────────────────────────────────────
# ANÁLISIS DE FORMA
# ─────────────────────────────────────────────────

def calcular_r2(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    """Coeficiente de determinación R²."""
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def analizar_forma(xs: np.ndarray, ys: np.ndarray) -> dict:
    """
    Extrae características de la curva para identificar
    qué familias son más probables.

    Retorna un dict con indicadores booleanos/numéricos
    y una lista ordenada de familias candidatas.
    """
    n = len(xs)
    forma = {}

    # ── Monotonía ──────────────────────────────────
    diffs = np.diff(ys)
    pct_pos = np.sum(diffs > 0) / len(diffs)
    forma["monotona_creciente"]  = pct_pos > 0.85
    forma["monotona_decreciente"] = pct_pos < 0.15
    forma["monotona"] = forma["monotona_creciente"] or forma["monotona_decreciente"]

    # ── Curvatura / segunda derivada ───────────────
    d2 = np.diff(diffs)
    forma["concava_arriba"]  = np.mean(d2) > 0.005 * (np.max(ys) - np.min(ys))
    forma["concava_abajo"]   = np.mean(d2) < -0.005 * (np.max(ys) - np.min(ys))
    forma["cambios_concavidad"] = np.sum(np.diff(np.sign(d2)) != 0)

    # ── Simetría ───────────────────────────────────
    mid = len(ys) // 2
    izq = ys[:mid]
    der = ys[mid:mid + len(izq)][::-1]
    forma["simetrica"] = np.corrcoef(izq, der)[0, 1] > 0.9 if len(izq) > 5 else False

    # ── Oscilación / periodicidad ──────────────────
    picos_max, _ = find_peaks(ys,  prominence=0.05 * (np.max(ys) - np.min(ys) + 1e-6))
    picos_min, _ = find_peaks(-ys, prominence=0.05 * (np.max(ys) - np.min(ys) + 1e-6))
    forma["num_picos"] = len(picos_max) + len(picos_min)
    forma["periodica"]  = forma["num_picos"] >= 3

    # Detectar periodo con FFT
    if forma["periodica"] and n > 20:
        fft_vals = np.abs(np.fft.rfft(ys - np.mean(ys)))
        frec_dom = np.argmax(fft_vals[1:]) + 1
        forma["periodo_estimado"] = float((xs[-1] - xs[0]) / frec_dom) if frec_dom > 0 else None
    else:
        forma["periodo_estimado"] = None

    # ── Rango y escala ─────────────────────────────
    forma["rango_y"] = float(np.max(ys) - np.min(ys))
    forma["media_y"] = float(np.mean(ys))
    forma["min_y"]   = float(np.min(ys))
    forma["max_y"]   = float(np.max(ys))

    # ── Detectar asíntota horizontal ──────────────
    extremo_izq = np.mean(ys[:max(1, n//10)])
    extremo_der = np.mean(ys[-(n//10):])
    forma["asintota_horizontal"] = abs(extremo_der - extremo_izq) < 0.05 * forma["rango_y"]

    # ── Clasificar familias candidatas ─────────────
    forma["familias_candidatas"] = _priorizar_familias(forma)

    return forma


def _priorizar_familias(f: dict) -> list[str]:
    """
    Devuelve la lista de familias ordenadas de más a
    menos probable según los indicadores de forma.
    """
    scores = {fam: 0 for fam in FAMILIAS}

    # Periódica → trigonométricas primero
    if f["periodica"]:
        scores["Trigonométrica"] += 10

    # Monotona + concavidad clara → exponencial o logarítmica
    if f["monotona_creciente"] and f["concava_arriba"]:
        scores["Exponencial"] += 8
    if f["monotona_creciente"] and f["concava_abajo"]:
        scores["Logarítmica"] += 8
        scores["Potencial"]    += 5
    if f["monotona_decreciente"] and f["concava_arriba"]:
        scores["Logarítmica"]  += 5
        scores["Exponencial"]  += 6

    # Sigmoide: monotona + asíntota horizontal
    if f["monotona"] and f["asintota_horizontal"]:
        scores["Racional"] += 9

    # Simétrica con un pico → gaussiana o cuadrática
    if f["simetrica"] and f["num_picos"] <= 2:
        scores["Exponencial"] += 5   # gaussiana
        scores["Polinómica"]  += 5

    # Cambios de curvatura → polinómica de mayor grado
    if f["cambios_concavidad"] >= 2:
        scores["Polinómica"]  += 7
    elif f["cambios_concavidad"] == 1:
        scores["Polinómica"]  += 4

    # Valores negativos de Y → descarta logarítmica directa
    if f["min_y"] < 0:
        scores["Logarítmica"] -= 3

    # Siempre añadir polinómica como fallback
    scores["Polinómica"] += 2

    return sorted(scores, key=lambda k: -scores[k])


# ─────────────────────────────────────────────────
# AJUSTE PARAMÉTRICO
# ─────────────────────────────────────────────────

@dataclass
class AjusteParcial:
    """Resultado del ajuste analítico en un tramo."""
    nombre      : str
    familia     : str
    latex       : str
    parametros  : dict
    r2          : float
    x_inicio    : float
    x_fin       : float
    funcion     : Callable
    xs_tramo    : np.ndarray
    ys_tramo    : np.ndarray


def ajustar_funcion(
    xs      : np.ndarray,
    ys      : np.ndarray,
    entrada : dict,
) -> Optional[AjusteParcial]:
    """
    Intenta ajustar UNA función del catálogo sobre los
    puntos dados. Prueba múltiples p0 para evitar mínimos
    locales. Devuelve AjusteParcial o None si falla.
    """
    func  = entrada["funcion"]
    p0    = list(entrada["p0"])
    latex = entrada["latex"]

    # Amplitudes heurísticas basadas en los datos
    amp_y   = float(np.max(ys) - np.min(ys)) + 1e-6
    amp_x   = float(np.max(xs) - np.min(xs)) + 1e-6
    mid_y   = float(np.mean(ys))
    mid_x   = float(np.mean(xs))

    # Generar varios puntos de inicio
    candidatos_p0 = [
        p0,
        [amp_y if abs(v) <= 1 else v for v in p0],
        [v * amp_y for v in p0],
    ]
    if len(p0) >= 2:
        candidatos_p0.append([amp_y, 1/amp_x if amp_x > 0 else 1] + p0[2:])
    if entrada["nombre"] in ("Seno", "Coseno") and len(p0) == 4:
        periodo_est = amp_x / max(1, len(find_peaks(ys)[0]))
        candidatos_p0.append([amp_y/2, 2*np.pi/max(periodo_est,0.1), 0.0, mid_y])

    mejor_r2     = -np.inf
    mejor_params = None

    for p_ini in candidatos_p0:
        try:
            params, _ = curve_fit(
                func, xs, ys,
                p0      = p_ini,
                maxfev  = 8000,
                bounds  = entrada.get("bounds", (-np.inf, np.inf)),
            )
            y_pred = func(xs, *params)
            r2 = calcular_r2(ys, y_pred)
            if r2 > mejor_r2:
                mejor_r2     = r2
                mejor_params = params
        except Exception:
            continue

    if mejor_params is None or mejor_r2 < -1.0:
        return None

    # Formatear LaTeX con parámetros reales
    param_names = ["a", "b", "c", "d", "e", "mu", "sigma"]
    slots = {param_names[i]: f"{mejor_params[i]:+.4g}"
             for i in range(len(mejor_params))}
    latex_fmt = latex
    for k, v in slots.items():
        latex_fmt = latex_fmt.replace("{" + k + "}", v)

    return AjusteParcial(
        nombre     = entrada["nombre"],
        familia    = entrada["familia"],
        latex      = latex_fmt,
        parametros = {param_names[i]: float(mejor_params[i])
                      for i in range(len(mejor_params))},
        r2         = mejor_r2,
        x_inicio   = float(xs[0]),
        x_fin      = float(xs[-1]),
        funcion    = func,
        xs_tramo   = xs,
        ys_tramo   = ys,
    )


def mejor_ajuste_para_tramo(
    xs        : np.ndarray,
    ys        : np.ndarray,
    verboso   : bool = False,
) -> Optional[AjusteParcial]:
    """
    Para un tramo (xs, ys):
      1. Analiza la forma
      2. Prueba funciones ordenadas por probabilidad
      3. Devuelve el mejor AjusteParcial encontrado
    """
    if len(xs) < MIN_PUNTOS:
        return None

    forma = analizar_forma(xs, ys)
    familias_orden = forma["familias_candidatas"]

    candidatos = []
    for familia in familias_orden:
        for entrada in FAMILIAS.get(familia, []):
            ajuste = ajustar_funcion(xs, ys, entrada)
            if ajuste is not None:
                candidatos.append(ajuste)
                if verboso:
                    print(f"    [{ajuste.familia}] {ajuste.nombre:30s}  R²={ajuste.r2:.4f}")

    if not candidatos:
        return None

    return max(candidatos, key=lambda a: a.r2)


# ─────────────────────────────────────────────────
# MOTOR RECURSIVO (divide y vencerás)
# ─────────────────────────────────────────────────

@dataclass
class ResultadoAnalitico:
    """Contiene todos los tramos con su función ajustada."""
    tramos      : list[AjusteParcial] = field(default_factory=list)
    profundidad : int = 0

    def r2_global(self) -> float:
        """R² promedio ponderado por longitud de tramo."""
        if not self.tramos:
            return 0.0
        total_n  = sum(len(t.xs_tramo) for t in self.tramos)
        r2_pond  = sum(t.r2 * len(t.xs_tramo) for t in self.tramos)
        return r2_pond / total_n

    def mostrar(self):
        """Imprime un resumen de los tramos."""
        print("\n" + "═"*55)
        print("  REPRESENTACIÓN ANALÍTICA")
        print("═"*55)
        for i, t in enumerate(self.tramos, 1):
            dominio = f"[{t.x_inicio:.3g}, {t.x_fin:.3g}]"
            print(f"\n  Tramo {i}/{len(self.tramos)}  {dominio}")
            print(f"  Familia   : {t.familia}")
            print(f"  Tipo      : {t.nombre}")
            print(f"  Expresión : f(x) = {t.latex}")
            print(f"  R²        : {t.r2:.4f}  {'✓ Bueno' if t.r2 >= R2_ACEPTABLE else '~ Aceptable' if t.r2 >= R2_TOLERABLE else '✗ Débil'}")
        print(f"\n  R² global ponderado : {self.r2_global():.4f}")
        print("═"*55)

    def graficar(self, titulo: str = "Representación Analítica"):
        """Visualiza los tramos junto con los puntos originales."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colores = cm.tab10.colors

        for i, t in enumerate(self.tramos):
            c = colores[i % len(colores)]
            xs_dense = np.linspace(t.x_inicio, t.x_fin, 500)
            ys_dense = t.funcion(xs_dense, *list(t.parametros.values()))

            ax.scatter(t.xs_tramo, t.ys_tramo, s=8, color=c, alpha=0.4, zorder=3)
            ax.plot(xs_dense, ys_dense, color=c, lw=2,
                    label=f"Tramo {i+1}: {t.nombre} (R²={t.r2:.3f})")

            # Divisor de tramo
            if i > 0:
                ax.axvline(t.x_inicio, color="gray", ls="--", lw=0.8, alpha=0.5)

        ax.set_title(f"{titulo}  —  R² global = {self.r2_global():.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig("representacion_analitica.png", dpi=150)
        plt.show()
        print("Gráfica guardada como representacion_analitica.png")


def analizar(
    puntos          : list[tuple[float, float]],
    umbral_r2       : float = R2_ACEPTABLE,
    profundidad_max : int   = MAX_PROFUNDIDAD,
    verboso         : bool  = True,
) -> ResultadoAnalitico:
    """
    Pipeline principal: puntos → representación analítica.

    Algoritmo recursivo (divide y vencerás):
      - Intenta ajustar el tramo completo
      - Si R² < umbral → lo divide en 2 y repite
      - Se detiene cuando R² es bueno o se alcanza
        la profundidad máxima

    Parámetros
    ----------
    puntos          : lista de (x, y) ordenada por x
    umbral_r2       : R² mínimo para aceptar un tramo
    profundidad_max : límite de divisiones recursivas
    verboso         : imprimir progreso en consola

    Retorna
    -------
    ResultadoAnalitico con todos los tramos ajustados.
    """
    xs = np.array([p[0] for p in puntos], dtype=float)
    ys = np.array([p[1] for p in puntos], dtype=float)

    resultado = ResultadoAnalitico()
    _ajustar_recursivo(xs, ys, resultado, umbral_r2, profundidad_max, 0, verboso)
    resultado.tramos.sort(key=lambda t: t.x_inicio)
    return resultado


def _ajustar_recursivo(
    xs        : np.ndarray,
    ys        : np.ndarray,
    resultado : ResultadoAnalitico,
    umbral_r2 : float,
    prof_max  : int,
    prof_actual: int,
    verboso   : bool,
):
    """Función interna recursiva."""
    indent = "  " * prof_actual
    rango_str = f"[{xs[0]:.3g}, {xs[-1]:.3g}]"

    if verboso:
        print(f"\n{indent}▶ Tramo {rango_str}  ({len(xs)} puntos, prof={prof_actual})")

    # Intentar ajustar este tramo
    ajuste = mejor_ajuste_para_tramo(xs, ys, verboso=verboso)

    if ajuste is None:
        if verboso:
            print(f"{indent}  ✗ No se encontró ajuste válido")
        return

    if verboso:
        print(f"{indent}  ★ Mejor: [{ajuste.familia}] {ajuste.nombre}  R²={ajuste.r2:.4f}")

    # ¿Es suficientemente bueno o llegamos al límite?
    if ajuste.r2 >= umbral_r2 or prof_actual >= prof_max or len(xs) < MIN_PUNTOS * 2:
        resultado.tramos.append(ajuste)
        resultado.profundidad = max(resultado.profundidad, prof_actual)
        if verboso:
            estado = "✓ ACEPTADO" if ajuste.r2 >= umbral_r2 else "⚠ LÍMITE ALCANZADO"
            print(f"{indent}  → {estado}")
        return

    # R² insuficiente → dividir en 2
    if verboso:
        print(f"{indent}  ↳ R²={ajuste.r2:.4f} < {umbral_r2} → dividiendo en 2...")

    mitad = len(xs) // 2
    xs_izq, ys_izq = xs[:mitad], ys[:mitad]
    xs_der, ys_der = xs[mitad:], ys[mitad:]

    _ajustar_recursivo(xs_izq, ys_izq, resultado, umbral_r2, prof_max, prof_actual + 1, verboso)
    _ajustar_recursivo(xs_der, ys_der, resultado, umbral_r2, prof_max, prof_actual + 1, verboso)


# ─────────────────────────────────────────────────
# EJEMPLO DE USO COMPLETO
# ─────────────────────────────────────────────────
if __name__ == "__main__":

    # --- Opción A: usar puntos generados por graph_extractor.py ---
    # from graph_extractor import extraer_puntos
    # puntos = extraer_puntos("mi_grafica.png", x_min=-5, x_max=5, y_min=-3, y_max=3)

    # --- Opción B: puntos sintéticos para prueba ---
    # Simulamos una curva mixta: seno + exponencial decreciente
    t = np.linspace(0, 4*np.pi, 300)
    y_test = 2 * np.sin(t) * np.exp(-0.15 * t) + 0.5

    puntos = [(float(xi), float(yi)) for xi, yi in zip(t, y_test)]
    print(f"Puntos de entrada: {len(puntos)}")

    # ── Ejecutar el motor analítico ────────────────
    resultado = analizar(
        puntos,
        umbral_r2       = R2_ACEPTABLE,
        profundidad_max = MAX_PROFUNDIDAD,
        verboso         = True,
    )

    # ── Mostrar resultados ─────────────────────────
    resultado.mostrar()

    # ── Graficar comparativa ───────────────────────
    resultado.graficar("Curva de prueba: 2·sin(t)·e^(-0.15t) + 0.5")
