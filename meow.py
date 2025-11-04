from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from pprint import pprint
type ArrayLike = list[float] | tuple[float, ...] | np.ndarray

def _err(msg: str) -> ValueError:
    return ValueError(f"мяу мяу, {msg}")

@dataclass(frozen=True)
class Vec:
    data: np.ndarray

    def __init__(self, data: ArrayLike):
        arr = np.asarray(data, dtype=float).reshape(-1)
        object.__setattr__(self, "data", arr)

    def __len__(self) -> int:
        return self.data.shape[0]

    def toarray(self) -> np.ndarray:
        return self.data.copy()

    def __add__(self, other: Vec) -> Vec:
        _check_vec_pair(self, other)
        return Vec(self.data + other.data)

    def __sub__(self, other: Vec) -> Vec:
        _check_vec_pair(self, other)
        return Vec(self.data - other.data)

    def __mul__(self, scalar: int | float) -> Vec:
        if isinstance(scalar, (int, float)):
            return Vec(self.data * scalar)
        raise _err("справа от вектора должен быть скаляр (int|float)")
    __rmul__ = __mul__

    def dot(self, other: Vec) -> float:
        _check_vec_pair(self, other)
        return float(self.data @ other.data)

    def norm(self, p: int | float = 2) -> float:
        if p == 1:
            return float(np.linalg.norm(self.data, ord=1))
        if p == 2:
            return float(np.linalg.norm(self.data))
        if p == np.inf:
            return float(np.linalg.norm(self.data, ord=np.inf))
        raise _err("норма вектора поддерживает только p из {1, 2, inf}")
    
    def tocolumn(self) -> np.ndarray:
        return self.data.reshape(-1, 1)

    def torow(self) -> np.ndarray:
        return self.data.reshape(1, -1)

@dataclass(frozen=True)
class Mat:
    data: np.ndarray

    def __init__(self, rows: ArrayLike | list[list[float]]):
        arr = np.asarray(rows, dtype=float)
        if arr.ndim != 2:
            raise _err("Mat ожидает 2D массив (n×m)")
        object.__setattr__(self, "data", arr)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    def toarray(self) -> np.ndarray:
        return self.data.copy()

    def is_square(self) -> bool:
        n, m = self.shape
        return n == m

    def is_symmetric(self, tol: float = 1e-12) -> bool:
        return self.is_square() and np.allclose(self.data, self.data.T, atol=tol, rtol=0)

    def __add__(self, other: Mat) -> Mat:
        _check_mat_pair(self, other)
        return Mat(self.data + other.data)

    def __sub__(self, other: Mat) -> Mat:
        _check_mat_pair(self, other)
        return Mat(self.data - other.data)

    def __mul__(self, scalar: int | float) -> Mat:
        if isinstance(scalar, (int, float)):
            return Mat(self.data * scalar)
        raise _err("справа от матрицы должен быть скаляр (int|float)")
    __rmul__ = __mul__

    # умножение матриц: A @ B  or  A @ v
    def __matmul__(self, other: Mat | Vec) -> Mat | Vec:
        if isinstance(other, Mat):
            return Mat(self.data @ other.data)
        if isinstance(other, Vec):
            return Vec(self.data @ other.data)
        raise _err("операция Mat @ ? поддерживает только Mat или Vec")

    @property
    def T(self) -> Mat:
        return Mat(self.data.T)

    def norm(self, kind: str = "2") -> float:
        k = kind.lower()
        if k == "1":
            return float(np.linalg.norm(self.data, ord=1))
        if k == "2":
            return float(np.linalg.norm(self.data, ord=2))
        if k in ("inf", "infty"):
            return float(np.linalg.norm(self.data, ord=np.inf))
        if k == "fro":
            return float(np.linalg.norm(self.data, ord="fro"))
        raise _err("вид нормы матрицы должен быть одним из {'1','2','inf','fro'}")

    def inv(self) -> Mat:
        if not self.is_square():
            raise _err("обратимая матрица должна быть квадратной")
        return Mat(np.linalg.inv(self.data))
    
    def det(self) -> float:
        if not self.is_square():
            raise _err("мяк, определитель определён только для квадратных матриц")
        return float(np.linalg.det(self.data))

    def cond(self, kind: str = "2") -> float:
        k = kind.lower()
        if k in {"inf", "infty"}:
            return float(np.linalg.cond(self.data, p=np.inf))
        if k == "1":
            return float(np.linalg.cond(self.data, p=1))
        if k == "2":
            return float(np.linalg.cond(self.data, p=2))
        raise _err("вид нормы для cond(A) должен быть '1', '2' или 'inf'")

    def _repr_latex_(self):
        try:
            import sympy as sp
            return r"$\displaystyle " + sp.latex(sp.Matrix(self.data)) + r"$"
        except Exception:
            return None
def eye(n: int) -> Mat:
    return Mat(np.eye(n, dtype=float))

def vec(x: ArrayLike) -> Vec:
    return Vec(x)

def mat(x: ArrayLike | list[list[float]]) -> Mat:
    return Mat(x)

def vector_norm(v: Vec, p: int | float = 2) -> float:
    return v.norm(p)

def matrix_norm(A: Mat, kind: str = "2") -> float:
    return A.norm(kind)

def mat_inv(A: Mat) -> Mat:
    return A.inv()

def cond_number(A: Mat, kind: str = "2") -> float:
    return A.cond(kind)

def _check_vec_pair(u: Vec, v: Vec) -> None:
    if not (isinstance(u, Vec) and isinstance(v, Vec) and len(u) == len(v)):
        raise _err("аргументы должны быть векторами одинаковой длины")

def _check_mat_pair(A: Mat, B: Mat) -> None:
    if not (isinstance(A, Mat) and isinstance(B, Mat) and A.shape == B.shape):
        raise _err("аргументы должны быть матрицами одинакового размера")

def _display_obj(obj):
    try:
        from IPython.display import display
    except Exception:
        def display(x):
            print(x)
    try:
        import sympy as sp
        HAS_SYMPY = True
    except Exception:
        HAS_SYMPY = False
    if isinstance(obj, Mat):
        if HAS_SYMPY:
            display(sp.Matrix(obj.data))
        else:
            display(obj.data)
    elif isinstance(obj, Vec):
        if HAS_SYMPY:
            display(sp.Matrix(obj.data.reshape(-1, 1)))
        else:
            display(obj.data.reshape(-1, 1))
    else:
        display(obj)

def _parse_floats_line(line: str) -> list[float]:
    toks = line.replace(",", " ").split()
    return [float(t) for t in toks]

def _read_vector(prompt: str = "введите элементы вектора через пробел/запятые: ") -> Vec:
    while True:
        try:
            xs = _parse_floats_line(input(prompt))
            if not xs:
                print("мяу мяу, вектор пустой - попробуйте ещё раз")
                continue
            return Vec(xs)
        except Exception as e:
            print(f"мяу мяу, не удалось распарсить вектор: {e}. попробуйте ещё разок")

def _read_matrix() -> Mat:
    while True:
        try:
            n = int(input("Введите число строк n: ").strip())
            m = int(input("Введите число столбцов m: ").strip())
            rows: list[list[float]] = []
            for i in range(n):
                line = input(f"Строка {i+1} - {m} чисел через пробел/запятые: ")
                row = _parse_floats_line(line)
                if len(row) != m:
                    print(f"мяу мяу, ожидалось {m} чисел, а получено {len(row)} - попробуем ещё раз.")
                    raise ValueError("bad row")
                rows.append(row)
            return Mat(rows)
        except Exception as e:
            print(f"мяу мяу, ошибка ввода матрицы: {e}\n")

def _to_sympy(obj):
    import sympy as sp
    import numpy as _np
    if isinstance(obj, Mat):
        return sp.Matrix(obj.data)
    if isinstance(obj, Vec):
        return sp.Matrix(obj.data.reshape(-1, 1))
    if isinstance(obj, (int, float, _np.floating)):
        return sp.Float(obj)
    if isinstance(obj, _np.ndarray):
        if obj.ndim == 1:
            return sp.Matrix(obj.reshape(-1, 1))
        if obj.ndim == 2:
            return sp.Matrix(obj)
    return sp.Symbol(str(obj))

def display_latex(obj, label: str | None = None):
    try:
        from IPython.display import display, Math
        import sympy as sp
        S = _to_sympy(obj)
        body = sp.latex(S)
        if label:
            display(Math(rf"{label} \;=\; {body}"))
        else:
            display(Math(rf"{body}"))
    except Exception:
        if label:
            print(f"{label} = {obj}")
        else:
            print(obj)