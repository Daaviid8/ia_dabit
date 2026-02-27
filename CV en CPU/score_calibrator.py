"""
====================================================
  SCORE CALIBRATOR  —  Etapa 2 del pipeline
====================================================
  Modela la relación empírica entre el score de
  salida de un clasificador Keras y la precisión
  real observada en validación.

  Flujo:
    1. Evaluar el modelo en el set de validación
    2. Construir la curva score → precisión real
    3. Ajustar una expresión analítica sobre esa curva
       (usando analytical_engine)
    4. Crear una capa Keras que calibra los scores
       usando esa función
    5. (Opcional) Ajustar umbral de decisión óptimo

  Dependencias:
    pip install tensorflow numpy scipy matplotlib

  Uso:
    from score_calibrator import CalibradorScores
    cal = CalibradorScores(modelo_keras)
    cal.ajustar(X_val, y_val)
    cal.mostrar()
    modelo_calibrado = cal.envolver_modelo()
====================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Motor analítico propio
from analytical_engine import analizar, ResultadoAnalitico

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_DISPONIBLE = True
except ImportError:
    TF_DISPONIBLE = False
    print("⚠ TensorFlow no instalado. Las capas Keras no estarán disponibles.")


# ════════════════════════════════════════════════════
#  PASO 1: GENERAR CURVA SCORE → PRECISIÓN REAL
# ════════════════════════════════════════════════════

def curva_score_precision(
    scores_pred : np.ndarray,   # scores del modelo [0, 1], shape (N,) o (N, C)
    y_true      : np.ndarray,   # etiquetas reales, shape (N,) o (N, C) one-hot
    n_bins      : int  = 40,
    clase       : Optional[int] = None,
    metodo      : str  = "isotonica",  # "bins" | "isotonica"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye la curva empírica score → precisión real.

    Para clasificación multiclase, toma la clase especificada
    o la clase con mayor score (one-vs-rest).

    Métodos:
        "bins"      : agrupa en n_bins y calcula precisión por bin
        "isotonica" : regresión isotónica (más suave y monotónica)

    Retorna (scores_eje, precisiones_eje) — arrays 1D ordenados.
    """
    # Aplanar a binario si multiclase
    if scores_pred.ndim == 2:
        if clase is not None:
            s = scores_pred[:, clase]
            y = (y_true == clase).astype(int) if y_true.ndim == 1 \
                else y_true[:, clase]
        else:
            # Macro: score máximo vs etiqueta correcta
            s = scores_pred.max(axis=1)
            y = (scores_pred.argmax(axis=1) ==
                 (y_true if y_true.ndim == 1 else y_true.argmax(axis=1))
                 ).astype(int)
    else:
        s = scores_pred.flatten()
        y = y_true.flatten()

    if metodo == "isotonica":
        return _curva_isotonica(s, y)
    else:
        return _curva_bins(s, y, n_bins)


def _curva_bins(s, y, n_bins):
    """Agrupa en bins y calcula precisión media por bin."""
    bins  = np.linspace(0, 1, n_bins + 1)
    centros, precisiones = [], []

    for i in range(n_bins):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if mask.sum() >= 5:
            centros.append((bins[i] + bins[i + 1]) / 2)
            precisiones.append(y[mask].mean())

    return np.array(centros), np.array(precisiones)


def _curva_isotonica(s, y):
    """
    Regresión isotónica: ajuste monotónico no paramétrico.
    Produce una curva más suave que los bins.
    """
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip")
    idx_orden = np.argsort(s)
    s_ord = s[idx_orden]
    y_ord = y[idx_orden].astype(float)
    p_ord = ir.fit_transform(s_ord, y_ord)

    # Muestrear 60 puntos representativos
    puntos_s   = np.linspace(s_ord.min(), s_ord.max(), 60)
    puntos_p   = ir.predict(puntos_s)
    return puntos_s, puntos_p


# ════════════════════════════════════════════════════
#  PASO 2: AJUSTE ANALÍTICO DE LA CURVA
# ════════════════════════════════════════════════════

def ajustar_curva_calibracion(
    scores_eje    : np.ndarray,
    precisiones_eje: np.ndarray,
    umbral_r2     : float = 0.95,
    verboso       : bool  = True,
) -> ResultadoAnalitico:
    """
    Aplica analytical_engine sobre la curva score → precisión
    para obtener una expresión analítica f(score) = precisión.

    Esta función es la que conecta el pipeline de extracción
    gráfica con la calibración del modelo.
    """
    puntos = [(float(s), float(p))
              for s, p in zip(scores_eje, precisiones_eje)]

    if verboso:
        print(f"  [Calibración] Ajustando {len(puntos)} puntos "
              f"de la curva score→precisión...")

    resultado = analizar(puntos, umbral_r2=umbral_r2, verboso=verboso)

    if verboso:
        print(f"  [Calibración] R² global: {resultado.r2_global():.4f}")

    return resultado


# ════════════════════════════════════════════════════
#  PASO 3: FUNCIÓN DE CALIBRACIÓN NUMPY
# ════════════════════════════════════════════════════

def construir_funcion_calibracion(
    resultado_analitico: ResultadoAnalitico,
) -> callable:
    """
    Construye una función Python f(score) → precisión_calibrada
    a partir del ResultadoAnalitico.

    Maneja múltiples tramos: evalúa cada score en el tramo
    cuyo dominio corresponde.
    """
    tramos = sorted(resultado_analitico.tramos, key=lambda t: t.x_inicio)

    def f_calibracion(score: np.ndarray) -> np.ndarray:
        score = np.atleast_1d(np.array(score, dtype=float))
        resultado = np.zeros_like(score)

        for tramo in tramos:
            mask = (score >= tramo.x_inicio) & (score <= tramo.x_fin)
            if mask.any():
                params = list(tramo.parametros.values())
                resultado[mask] = tramo.funcion(score[mask], *params)

        # Clip a [0, 1] — es una precisión
        return np.clip(resultado, 0.0, 1.0)

    return f_calibracion


# ════════════════════════════════════════════════════
#  PASO 4: CAPA KERAS DE CALIBRACIÓN
# ════════════════════════════════════════════════════

if TF_DISPONIBLE:

    class CapaCalibrador(keras.layers.Layer):
        """
        Capa Keras que aplica la calibración analítica
        sobre los scores de salida del clasificador.

        Puede insertarse directamente sobre la capa
        de salida (softmax) del modelo existente.

        Parámetros de la curva: almacenados como pesos
        no entrenables → la calibración es fija (derivada
        del ajuste analítico) y no se modifica durante
        fine-tuning salvo que se indique explícitamente.
        """

        def __init__(self, funcion_calibracion: callable,
                     nombre="calibrador_analitico", **kwargs):
            super().__init__(name=nombre, **kwargs)
            self._fn_cal = funcion_calibracion

        def call(self, inputs):
            """
            Aplica f_calibracion elemento a elemento.
            Usa tf.py_function para envolver la función numpy.
            """
            def _aplicar(x):
                x_np = x.numpy()
                cal  = self._fn_cal(x_np)
                # Renormalizar para que sume 1 en multiclase
                total = cal.sum(axis=-1, keepdims=True) + 1e-10
                return (cal / total).astype(np.float32)

            return tf.py_function(
                func=_aplicar,
                inp=[inputs],
                Tout=tf.float32,
            )

        def get_config(self):
            # Nota: la función no es serializable directamente.
            # Para guardar el modelo, usar guardar_calibrador().
            return super().get_config()


# ════════════════════════════════════════════════════
#  UMBRAL DE DECISIÓN ÓPTIMO
# ════════════════════════════════════════════════════

def umbral_optimo(
    f_calibracion : callable,
    metrica       : str   = "f1",       # "f1" | "precision" | "recall"
    target_precision: Optional[float] = None,  # ej: 0.90
    n_puntos      : int   = 1000,
) -> dict:
    """
    Calcula el umbral de decisión óptimo dado la función
    de calibración analítica.

    Si target_precision se especifica, devuelve el umbral
    mínimo que garantiza esa precisión.

    Si metrica="f1", maximiza el F1-score teórico.

    Retorna dict con umbral, precisión esperada, recall estimado.
    """
    umbrales    = np.linspace(0.01, 0.99, n_puntos)
    precisiones = f_calibracion(umbrales)

    if target_precision is not None:
        # Umbral mínimo que garantiza target_precision
        candidatos = umbrales[precisiones >= target_precision]
        if len(candidatos) == 0:
            print(f"  ⚠ Ningún umbral garantiza precisión ≥ {target_precision}")
            umbral_sel = 0.5
        else:
            umbral_sel = float(candidatos.min())
        return {
            "umbral"   : umbral_sel,
            "precision": float(f_calibracion(np.array([umbral_sel]))[0]),
            "criterio" : f"target_precision={target_precision}",
        }

    if metrica == "f1":
        # F1 = 2·P·R / (P+R), asumiendo recall ≈ 1 - umbral (heurística)
        recalls_est = 1.0 - umbrales
        f1_scores   = (2 * precisiones * recalls_est /
                       (precisiones + recalls_est + 1e-10))
        idx = np.argmax(f1_scores)
    elif metrica == "precision":
        idx = np.argmax(precisiones)
    else:
        idx = len(umbrales) // 2   # 0.5 por defecto

    return {
        "umbral"   : float(umbrales[idx]),
        "precision": float(precisiones[idx]),
        "criterio" : metrica,
    }


# ════════════════════════════════════════════════════
#  CLASE PRINCIPAL: CalibradorScores
# ════════════════════════════════════════════════════

class CalibradorScores:
    """
    Interfaz unificada para todo el pipeline de calibración.

    Uso típico:
        cal = CalibradorScores(modelo)
        cal.ajustar(X_val, y_val)
        cal.mostrar()
        modelo_cal = cal.envolver_modelo()
        umbral     = cal.umbral_optimo(target_precision=0.90)
    """

    def __init__(self, modelo=None, n_bins=40, umbral_r2=0.95,
                 metodo_curva="isotonica"):
        self.modelo        = modelo
        self.n_bins        = n_bins
        self.umbral_r2     = umbral_r2
        self.metodo_curva  = metodo_curva

        self.scores_eje    = None
        self.precisiones_eje = None
        self.resultado_analitico = None
        self.f_calibracion = None

    def ajustar(self, X_val, y_val, clase=None, batch_size=64):
        """
        Evalúa el modelo sobre (X_val, y_val), construye la
        curva score→precisión y ajusta la expresión analítica.
        """
        SEP = "─" * 50
        print(f"\n{SEP}\n  CALIBRADOR DE SCORES\n{SEP}")

        # Obtener scores del modelo
        if self.modelo is not None and TF_DISPONIBLE:
            print("[1/3] Evaluando modelo en validación...")
            scores = self.modelo.predict(X_val, batch_size=batch_size,
                                         verbose=0)
        else:
            print("[1/3] Sin modelo → usando X_val directamente como scores")
            scores = np.array(X_val)

        # Curva empírica
        print("[2/3] Construyendo curva score→precisión...")
        self.scores_eje, self.precisiones_eje = curva_score_precision(
            scores, np.array(y_val),
            n_bins=self.n_bins,
            clase=clase,
            metodo=self.metodo_curva,
        )
        print(f"  Rango scores: [{self.scores_eje.min():.3f}, "
              f"{self.scores_eje.max():.3f}]")
        print(f"  Rango precisión: [{self.precisiones_eje.min():.3f}, "
              f"{self.precisiones_eje.max():.3f}]")

        # Ajuste analítico
        print("[3/3] Ajuste analítico de la curva...")
        self.resultado_analitico = ajustar_curva_calibracion(
            self.scores_eje, self.precisiones_eje,
            umbral_r2=self.umbral_r2, verboso=True,
        )
        self.f_calibracion = construir_funcion_calibracion(
            self.resultado_analitico)

        print(f"\n  ✓ Calibrador listo\n{SEP}")
        return self

    def mostrar(self):
        """Imprime la expresión analítica y grafica la calibración."""
        if self.resultado_analitico is None:
            print("⚠ Llama primero a .ajustar()")
            return

        self.resultado_analitico.mostrar()
        self._graficar()

    def _graficar(self, ruta="calibracion_scores.png"):
        """Gráfica: curva empírica vs ajuste analítico."""
        if self.scores_eje is None:
            return

        s_dense = np.linspace(self.scores_eje.min(),
                               self.scores_eje.max(), 500)
        p_ajust = self.f_calibracion(s_dense)

        plt.figure(figsize=(8, 5))
        plt.scatter(self.scores_eje, self.precisiones_eje,
                    c="gray", s=20, alpha=0.7, label="Empírica (validación)")
        plt.plot(s_dense, p_ajust, "b-", lw=2,
                 label=f"Ajuste analítico (R²={self.resultado_analitico.r2_global():.4f})")
        plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Calibración perfecta")
        plt.xlabel("Score del modelo"); plt.ylabel("Precisión real")
        plt.title("Calibración de scores: empírica vs analítica")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ruta, dpi=150)
        plt.close()
        print(f"  Gráfica guardada: {ruta}")

    def umbral_optimo(self, metrica="f1", target_precision=None):
        """Calcula el umbral de decisión óptimo."""
        if self.f_calibracion is None:
            print("⚠ Llama primero a .ajustar()")
            return None
        resultado = umbral_optimo(self.f_calibracion, metrica,
                                   target_precision)
        print(f"\n  Umbral óptimo ({resultado['criterio']}): "
              f"{resultado['umbral']:.4f}  "
              f"→ precisión esperada: {resultado['precision']:.4f}")
        return resultado

    def envolver_modelo(self):
        """
        Devuelve un nuevo modelo Keras con la capa de
        calibración añadida sobre la salida del modelo original.
        """
        if not TF_DISPONIBLE:
            print("⚠ TensorFlow no disponible")
            return None
        if self.modelo is None:
            print("⚠ No hay modelo base")
            return None
        if self.f_calibracion is None:
            print("⚠ Llama primero a .ajustar()")
            return None

        capa_cal = CapaCalibrador(self.f_calibracion)
        entrada  = self.modelo.input
        salida_cal = capa_cal(self.modelo.output)
        modelo_cal = keras.Model(inputs=entrada, outputs=salida_cal,
                                  name=self.modelo.name + "_calibrado")
        print(f"  Modelo calibrado creado: {modelo_cal.name}")
        return modelo_cal

    def calibrar_scores(self, scores: np.ndarray) -> np.ndarray:
        """Aplica la función de calibración directamente sobre scores."""
        if self.f_calibracion is None:
            raise RuntimeError("Llama primero a .ajustar()")
        return self.f_calibracion(scores)


# ════════════════════════════════════════════════════
#  EJEMPLO DE USO
# ════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Ejemplo con datos sintéticos (sin modelo real) ──
    np.random.seed(42)
    n = 2000

    # Simular scores de un clasificador imperfecto
    y_true   = np.random.randint(0, 2, n)
    # Scores bien calibrados añadiendo ruido
    scores_raw = y_true * 0.6 + np.random.beta(2, 5, n) * 0.5
    scores_raw = np.clip(scores_raw, 0, 1)

    cal = CalibradorScores(modelo=None, umbral_r2=0.93,
                           metodo_curva="isotonica")
    cal.ajustar(scores_raw, y_true)
    cal.mostrar()

    info_umbral = cal.umbral_optimo(target_precision=0.85)
    print(f"\nUsar umbral={info_umbral['umbral']:.3f} "
          f"para garantizar precisión ≥ 0.85")

    # ── Ejemplo con modelo Keras real ──────────────────
    # modelo = keras.models.load_model("mi_clasificador.h5")
    # cal = CalibradorScores(modelo, umbral_r2=0.95)
    # cal.ajustar(X_val, y_val)
    # cal.mostrar()
    # modelo_calibrado = cal.envolver_modelo()
    # modelo_calibrado.save("clasificador_calibrado.h5")
