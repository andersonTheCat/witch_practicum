import numpy as np
from IPython.display import display, Markdown
from meow import (
    Mat, Vec, display_latex, _read_matrix, _read_vector,
    _display_obj, _err, eye, vec, mat
)
def _as_array_M(obj) -> np.ndarray:
    if isinstance(obj, Mat):
        return obj.data.astype(float, copy=True)
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 2:
        raise _err("ожидалась матрица (2D)")
    return arr.copy()
def _as_array_v(obj):
    if isinstance(obj, Vec):
        return obj.data.astype(float, copy=True).reshape(-1)
    return np.asarray(obj, dtype=float).reshape(-1).copy()
def display_math(tex: str):
    try:
        from IPython.display import Math
        display(Math(tex))
    except Exception:
        print(tex)
def _parse_pivot(s: str):
    if not s:
        return "col"
    s = s.strip().lower()
    trans = str.maketrans({
        "с":"c","о":"o","е":"e","р":"p","а":"a","х":"x",
        "к":"k","м":"m","т":"t","н":"h","у":"y","в":"b","л":"l"
    })
    s = s.translate(trans)
    aliases = {
        "col":"col","c":"col","column":"col","ст":"col","столбец":"col",
        "row":"row","r":"row","стр":"row","строка":"row",
        "full":"full","f":"full","полный":"full","полн":"full",
        "none":"none","n":"none","без":"none","нет":"none"
    }
    return aliases.get(s, "col")
def gauss_ops_theory(n: int):
    Qf = (2/3)*n**3 + 0.5*n**2 - (7/6)*n
    Qb = n**2
    return {"forward": Qf, "backward": Qb, "total": Qf + Qb}
def gauss_solve(A_in, b_in, pivot: str = "col"):
    A = _as_array_M(A_in)
    b = _as_array_v(b_in)
    n, m = A.shape
    if n != m:
        raise _err("метод Гаусса кушает только квадратные матрицы A")
    if b.shape != (n,):
        raise _err("размерность правой части b должна быть (n,)")

    ops = {"add": 0, "sub": 0, "mul": 0, "div": 0}
    def _mul(x, y): ops["mul"] += 1; return x*y
    def _div(x, y): ops["div"] += 1; return x/y
    def _sub(x, y): ops["sub"] += 1; return x-y
    def _add(x, y): ops["add"] += 1; return x+y

    row_swaps = 0
    col_swaps = 0
    col_perm = list(range(n))

    for k in range(n-1):
        ik, jk = k, k
        if pivot == "col":
            i_rel = int(np.argmax(np.abs(A[k:, k])))
            ik = k + i_rel
        elif pivot == "row":
            j_rel = int(np.argmax(np.abs(A[k, k:])))
            jk = k + j_rel
        elif pivot == "full":
            i_rel, j_rel = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
            ik, jk = k + int(i_rel), k + int(j_rel)
        elif pivot == "none":
            pass
        else:
            raise _err("здесь живут только такие способы выбора ведущего элемента: {'none','col','row','full'}")

        if ik != k:
            A[[k, ik], :] = A[[ik, k], :]
            b[k], b[ik] = b[ik], b[k]
            row_swaps += 1
        if jk != k:
            A[:, [k, jk]] = A[:, [jk, k]]
            col_perm[k], col_perm[jk] = col_perm[jk], col_perm[k]
            col_swaps += 1

        piv = A[k, k]
        if piv == 0.0:
            raise _err("мяу мяу, встретился нулевой ведущий элемент - попробуйте другой режим выбора")

        for i in range(k+1, n):
            aik = A[i, k]
            if aik == 0.0:
                continue
            mu = _div(aik, piv)
            A[i, k] = 0.0
            for j in range(k+1, n):
                A[i, j] = _sub(A[i, j], _mul(mu, A[k, j]))
            b[i] = _sub(b[i], _mul(mu, b[k]))

    det = float(np.prod(np.diag(A)))
    if (row_swaps + col_swaps) % 2 == 1:
        det = -det

    diag = np.diag(A)
    if np.any(diag == 0.0):
        for i in range(n):
            if np.all(A[i, :] == 0.0) and not np.isclose(b[i], 0.0):
                raise _err("система несовместна: строка 0...0, но b_i ≠ 0")
        raise _err("матрица вырождена: нулевой диагональный элемент в U")

    x_perm = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        s = b[i]
        for j in range(i+1, n):
            s = _sub(s, _mul(A[i, j], x_perm[j]))
        x_perm[i] = _div(s, A[i, i])

    x = np.zeros_like(x_perm)
    for i, var_idx in enumerate(col_perm):
        x[var_idx] = x_perm[i]

    ops["total"] = sum(ops.values())

    info = {
        "U": Mat(A),
        "b_mod": Vec(b),
        "det": det,
        "swaps": {"row": row_swaps, "col": col_swaps},
        "col_perm": col_perm,
        "ops": ops,
    }
    return Vec(x), info
def pre_assess(A):
    Aarr = _as_array_M(A)
    det_val = A.det() if isinstance(A, Mat) else float(np.linalg.det(Aarr))
    return {
        "shape": Aarr.shape,
        "cond2": float(np.linalg.cond(Aarr, 2)),
        "det":   det_val,
    }
def kek(n: int):
    """A_ij = 1/(i+j-1)"""
    i = np.arange(1, n+1)[:, None]
    j = np.arange(1, n+1)[None, :]
    return Mat(1.0 / (i + j - 1.0))
def add_diag(A_in, eps: float):
    A = _as_array_M(A_in).copy()
    idx = np.arange(A.shape[0])
    A[idx, idx] += eps
    return Mat(A)
def _maybe_log(name, k, L=None, U=None, P=None, log_every=None, checkpoints=None):
    if log_every is None and not checkpoints:
        return
    hit = False
    if log_every is not None and log_every > 0 and (k % log_every == 0):
        hit = True
    if checkpoints and k in checkpoints:
        hit = True
    if not hit:
        return
    display(Markdown(f"**{name}: шаг k = {k}**"))
    if P is not None:  display_latex(Mat(P), label=r"P")
    if L is not None:  display_latex(Mat(L), label=r"L")
    if U is not None:  display_latex(Mat(U), label=r"U")
def _ops_zero():
    return {"add":0,"sub":0,"mul":0,"div":0,"total":0}
def _ops_inc(ops, key, k=1):
    ops[key] += k; ops["total"] += k
def _ops_merge(*many):
    out = _ops_zero()
    for d in many:
        for k in out: out[k] += d.get(k,0)
    return out
def lu_nopivot(A_in, log_every=None, checkpoints=None):
    A = _as_array_M(A_in); n = A.shape[0]
    L = np.eye(n); U = A.copy()
    ops = _ops_zero()
    _maybe_log("LU", 0, L=L, U=U, log_every=log_every, checkpoints=checkpoints)
    for k in range(n-1):
        piv = U[k,k]
        if piv == 0.0:
            raise _err("нулевой ведущий элемент в LU без перестановок")
        for i in range(k+1, n):
            L[i,k] = U[i,k] / piv; _ops_inc(ops,"div")
            U[i,k] = 0.0
            for j in range(k+1, n):
                U[i,j] -= L[i,k]*U[k,j]; _ops_inc(ops,"mul"); _ops_inc(ops,"sub")
        _maybe_log("LU", k+1, L=L, U=U, log_every=log_every, checkpoints=checkpoints)
    return Mat(L), Mat(U), ops
def lup(A_in, log_every=None, checkpoints=None):
    A = _as_array_M(A_in); n = A.shape[0]
    P = np.eye(n); L = np.eye(n); U = A.copy()
    swaps = 0; ops = _ops_zero()
    _maybe_log("LUP", 0, P=P, L=L, U=U, log_every=log_every, checkpoints=checkpoints)
    for k in range(n-1):
        p = k + int(np.argmax(np.abs(U[k:,k])))
        if abs(U[p,k]) == 0.0:
            raise _err("ранг(A) < n: столбец нулевой ниже диагонали")
        if p != k:
            U[[k,p],:] = U[[p,k],:]
            P[[k,p],:] = P[[p,k],:]
            if k>0: L[[k,p],:k] = L[[p,k],:k]
            swaps += 1
        piv = U[k,k]
        for i in range(k+1, n):
            L[i,k] = U[i,k] / piv; _ops_inc(ops,"div")
            U[i,k] = 0.0
            for j in range(k+1, n):
                U[i,j] -= L[i,k]*U[k,j]; _ops_inc(ops,"mul"); _ops_inc(ops,"sub")
        _maybe_log("LUP", k+1, P=P, L=L, U=U, log_every=log_every, checkpoints=checkpoints)
    return Mat(P), Mat(L), Mat(U), swaps, ops
def forward_subst(L_in, b_in):
    L = _as_array_M(L_in); b = _as_array_v(b_in)
    n = L.shape[0]; y = np.zeros(n); ops = _ops_zero()
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i,j]*y[j]; _ops_inc(ops,"mul"); _ops_inc(ops,"sub")
        y[i] = s
    return y, ops
def backward_subst(U_in, y_in):
    U = _as_array_M(U_in); y = _as_array_v(y_in)
    n = U.shape[0]; x = np.zeros(n); ops = _ops_zero()
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= U[i,j]*x[j]; _ops_inc(ops,"mul"); _ops_inc(ops,"sub")
        if U[i,i] == 0.0: raise _err("деление на ноль в обратном ходе (U вырождена)")
        x[i] = s / U[i,i]; _ops_inc(ops,"div")
    return x, ops
def solve_via_LU(L, U, b):
    y, of = forward_subst(L, b)
    x, ob = backward_subst(U, y)
    return Vec(x), _ops_merge(of, ob)
def solve_via_LUP(P, L, U, b):
    y, of = forward_subst(L, P.data @ _as_array_v(b))
    x, ob = backward_subst(U, y)
    return Vec(x), _ops_merge(of, ob)
def det_from_U(U_in, swaps: int = 0):
    U = _as_array_M(U_in)
    det = float(np.prod(np.diag(U)))
    return -det if (swaps % 2 == 1) else det
def run_many_rhs_uniform(A_in, m: int, pivot: str = "col", a: float = -10.0, b: float = 10.0, seed: int | None = 42):
    rng = np.random.default_rng(seed)
    A = _as_array_M(A_in); n = A.shape[0]
    X_true = rng.uniform(a, b, size=(n, m))
    B = A @ X_true

    rel_res_all, rel_err_all = [], []
    ops_total = _ops_zero()

    for j in range(m):
        x_hat, info = gauss_solve(A_in, Vec(B[:,j]), pivot=pivot)
        xh = x_hat.data
        r = A @ xh - B[:,j]
        rel_res_all.append(np.linalg.norm(r,2) / (np.linalg.norm(A,2)*np.linalg.norm(xh,2)))
        rel_err_all.append(np.linalg.norm(xh - X_true[:,j],2) / np.linalg.norm(X_true[:,j],2))
        ops_total = _ops_merge(ops_total, info["ops"])

    display(Markdown(f"**серия решений Гауссом:** m=`{m}`, pivot=`{pivot}`, x~`U[{a},{b}]`"))
    display_latex(np.mean(rel_res_all), label=r"\mathrm{mean}\ \mathrm{rel\_res}")
    display_latex(np.max(rel_res_all),  label=r"\mathrm{max}\ \mathrm{rel\_res}")
    display_latex(np.mean(rel_err_all), label=r"\mathrm{mean}\ \mathrm{rel\_err}")
    display_latex(np.max(rel_err_all),  label=r"\mathrm{max}\ \mathrm{rel\_err}")
    display_latex(ops_total["total"],   label=r"Q_{\mathrm{meas}}\ \text{(вся серия)}")
    return {"ops_total": ops_total, "rel_res": rel_res_all, "rel_err": rel_err_all}
def compare_ops(A_in, m: int, pivot_for_gauss: str = "col", a=-10.0, b=10.0, seed=42):
    rng = np.random.default_rng(seed)
    A = _as_array_M(A_in); n = A.shape[0]
    X_true = rng.uniform(a,b,size=(n,m)); B = A @ X_true

    P, Lp, Up, swaps, ops_fact = lup(A_in)
    ops_series_lup = _ops_zero()
    for j in range(m):
        _, ops_solve = solve_via_LUP(P, Lp, Up, Vec(B[:,j]))
        ops_series_lup = _ops_merge(ops_series_lup, ops_solve)
    total_lup = _ops_merge(ops_fact, ops_series_lup)

    ops_gauss_all = _ops_zero()
    for j in range(m):
        _, info = gauss_solve(A_in, Vec(B[:,j]), pivot=pivot_for_gauss)
        ops_gauss_all = _ops_merge(ops_gauss_all, info["ops"])

    th = gauss_ops_theory(n)
    th_lup_fact  = (2/3)*n**3
    th_tri_rhs   = 2*n**2
    th_lup_total = th_lup_fact + m*th_tri_rhs

    display(Markdown("### сравнение числа операций"))
    display(Markdown(f"- размер: `n={n}`, правых частей: `m={m}`"))
    display(Markdown("**LUP:** одно разложение матрицы A + m решений"))
    display(Markdown(f"- разложение A: `Q = {ops_fact['total']}`"))
    display(Markdown(f"- решения (в сумме): `Q = {ops_series_lup['total']}`"))
    display(Markdown(f"- измерено всего (LUP): `Q = {total_lup['total']}`"))

    display(Markdown("**Гаусс:** решаем m раз с нуля"))
    display(Markdown(f"- измерено всего (Гаусс): `Q = {ops_gauss_all['total']}`"))

    return {
        "ops_lup_fact": ops_fact, 
        "ops_lup_solve_sum": ops_series_lup,
        "ops_lup_total": total_lup,
        "ops_gauss_total": ops_gauss_all,
        "theory": {
            "gauss_one": th["total"],
            "gauss_m": m*th["total"],
            "lup_fact": th_lup_fact,
            "tri_per_rhs": th_tri_rhs,
            "lup_total": th_lup_total
        }
    }
def _rand_orth(n, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q
def make_random_A(n: int, cond_target: float = 1e3, seed: int | None = None):
    rng = np.random.default_rng(seed)
    Q1 = _rand_orth(n, rng); Q2 = _rand_orth(n, rng)
    if cond_target < 1.0: cond_target = 1.0
    logs = rng.uniform(0.0, np.log(cond_target), size=n)
    svals = np.exp(logs)
    A = (Q1 @ np.diag(svals) @ Q2.T).astype(float)
    return Mat(A)
def make_random_system(n: int, a: float = -10.0, b: float = 10.0,
                       cond_target: float = 1e3, seed: int | None = 42):
    A = make_random_A(n, cond_target=cond_target, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 1)
    x_true = rng.uniform(a, b, size=n)
    b_vec = A.data @ x_true
    return A, Vec(x_true), Vec(b_vec)
def make_random_system_simple(n: int, a: float = -10.0, b: float = 10.0,
                              seed: int | None = 42, spd: bool = False,
                              show: bool = True):
    rng = np.random.default_rng(seed)
    if spd:
        G = rng.standard_normal((n, n))
        A = (G.T @ G).astype(float)
    else:
        A = rng.standard_normal((n, n)).astype(float)

    x_true = rng.uniform(a, b, size=n)
    b_vec  = A @ x_true

    if show:
        msg_spd = ", симметричная положительно определённая" if spd else ""
        display(Markdown(
            rf"случайная матрица $A${msg_spd}; сгенерирован точный "
            rf"$x_{{\rm true}}\sim\mathcal U[{a},{b}]$ (seed={seed}):"
        ))
        display_latex(Vec(x_true), label=r"x_{\mathrm{true}}")

    return Mat(A), Vec(x_true), Vec(b_vec)
def _read_square_A_and_b():
    try:
        n = int(input("введите n (размер квадратной матрицы A): ").strip())
        if n <= 0: raise ValueError
    except Exception:
        raise _err("мяу мяу, n должно быть положительным целым")

    rows = []
    print("введите построчно элементы A (через пробел):")
    for i in range(n):
        line = input(f"строка {i+1}: ").strip()
        try:
            row = [float(x) for x in line.replace(",", " ").split()]
        except Exception:
            raise _err("не удалось распарсить числа для A :(")
        if len(row) != n:
            raise _err(f"ожидалось {n} чисел в строке {i+1}")
        rows.append(row)
    A = Mat(np.array(rows, dtype=float))

    line = input("введите b (n чисел через пробел): ").strip()
    try:
        b_vals = [float(x) for x in line.replace(",", " ").split()]
    except Exception:
        raise _err("не удалось распарсить числа для b :(")
    if len(b_vals) != A.shape[0]:
        raise _err(f"ожидалось {A.shape[0]} чисел для b")
    b = Vec(np.array(b_vals, dtype=float))
    return A, b
def main():
    display(Markdown(
        "мяу мяу, что делаем?\n\n"
        "[1] решить $A x=b$ (ввод вручную / случайно)\n\n"
        "[2] решить $A x=b$ для Гильберта: $A_{ij}=\\frac{1}{i+j-1},\\ x=\\mathbf{1}$ (с опцией $+\\varepsilon I$)\n\n"
        "[3] LU-разложение ($A=L\\,U$) и, по желанию, решение\n\n"
        "[4] LUP-разложение ($P A=L\\,U$) и, по желанию, решение\n\n"
        "[5] серия правых частей по (38): $x\\sim\\mathcal U[-10,10],\\ b=A x$\n"
    ))

    mode = input("ваш выбор (1/2/3/4/5): ").strip()

    if mode == "1":
        how = (input("как задаём A и b? (m=вручную / r=случайно / rc=случайно с заданной условностью): ")
               .strip().lower() or "m")

        if how.startswith("r") and not how.startswith("rc"):
            n = int(input("введите n: ").strip())
            try:
                a = float(input("нижняя граница для x_true a [по умолчанию -10]: ").strip() or "-10")
                b_rng = float(input("верхняя граница для x_true b [по умолчанию 10]: ").strip() or "10")
            except Exception:
                a, b_rng = -10.0, 10.0
            spd_ans = (input("делать SPD (A=G^T G)? (y/n, по умолчанию n): ").strip().lower() or "n")
            spd = spd_ans.startswith("y")
            try:
                seed = int(input("seed (целое, пусто = 42): ").strip() or "42")
            except Exception:
                seed = 42

            A, x_true, b = make_random_system_simple(n, a=a, b=b_rng, seed=seed, spd=spd, show=True)
            msg_spd = ", симметричная положительно определённая" if spd else ""
            display(Markdown(
                rf"случайная $A${msg_spd}, "
                rf"$x_{{\rm true}}\sim\mathcal U[{a},{b_rng}]$, затем формируем $b=A\,x_{{\rm true}}$"
            ))
        elif how.startswith("rc"):
            n = int(input("введите n: ").strip())
            try:
                cond_target = float(input("желаемая условность A (≈cond), напр. 1e3 [по умолчанию]: ").strip() or "1e3")
            except Exception:
                cond_target = 1e3
            try:
                a = float(input("нижняя граница для x_true a [по умолчанию -10]: ").strip() or "-10")
                b_rng = float(input("верхняя граница для x_true b [по умолчанию 10]: ").strip() or "10")
            except Exception:
                a, b_rng = -10.0, 10.0
            try:
                seed = int(input("seed (целое, пусто = 42): ").strip() or "42")
            except Exception:
                seed = 42

            A, x_true, b = make_random_system(n, a=a, b=b_rng, cond_target=cond_target, seed=seed)
            display(Markdown(
                rf"случайная $A$ (ориентировочно $\kappa\!\approx\!{cond_target:g}$), "
                rf"$x_{{\rm true}}\sim\mathcal U[{a},{b_rng}]$, затем $b=A\,x_{{\rm true}}$"
            ))
        else:
            A, b = _read_square_A_and_b()
            x_true = None

    elif mode == "2":
        n = int(input("введите n: ").strip())
        A_orig = kek(n)
        x_true = Vec([1.0]*n)
        b = Vec(A_orig.data @ x_true.data)

        use_shift = (input("добавить ε⋅I к диагонали? (y/n, по умолчанию n): ").strip().lower() or "n").startswith("y")
        if use_shift:
            try:
                eps = float(input("ε = (например 1e-10): ").strip() or "1e-10")
            except Exception:
                eps = 1e-10
            A_reg = add_diag(A_orig, eps)
            refit_b = (input("пересчитать b под A+εI (b:=(A+εI)·1)? (y/n, по умолчанию y): ")
                        .strip().lower() or "y").startswith("y")
            if refit_b:
                b = Vec(A_reg.data @ x_true.data)
                display(Markdown(rf"регуляризация: решаем $(A+\varepsilon I)x=b$ с $\varepsilon={eps:.3g}$, где $b=(A+\varepsilon I)\,\mathbf 1$"))
            else:
                display(Markdown(rf"регуляризация: решаем $(A+\varepsilon I)x=b$ с $\varepsilon={eps:.3g}$, где $b=A\,\mathbf 1$ (исходный)"))
            A = A_reg
        else:
            A = A_orig
        display(Markdown(r"точное решение задаём как $x=\mathbf{1}$"))

    elif mode in {"3","4","5"}:
        howA = input("как задаём A?  (m=вручную / h=Гильберт): ").strip().lower() or "m"
        if howA == "m":
            try:
                n = int(input("введите n (размер квадратной матрицы A): ").strip())
                if n <= 0: raise ValueError
            except Exception:
                raise _err("мяу мяу, n должно быть положительным целым")
            rows = []
            print("введите построчно элементы A (через пробел):")
            for i in range(n):
                line = input(f"строка {i+1}: ").strip()
                row = [float(x) for x in line.replace(",", " ").split()]
                if len(row) != n: raise _err(f"ожидалось {n} чисел в строке {i+1}")
                rows.append(row)
            A = Mat(np.array(rows, dtype=float))
        else:
            n = int(input("введите n: ").strip())
            A = kek(n)
        b = None; x_true = None
    else:
        print("мяу мяу, нужно выбрать 1, 2, 3, 4 или 5")
        return

    stats = pre_assess(A)
    display(Markdown(r"всякое:"))
    display_latex(A,               label=r"A")
    display_latex(stats["cond2"],  label=r"\mathrm{cond}_2(A)")
    display_latex(stats["det"],    label=r"\det(A)")

    if mode in {"1","2"}:
        display(Markdown("выбор главного элемента: `none` / `col` / `row` / `full`"))
        pivot = _parse_pivot(input("pivot → "))
        x_hat, info = gauss_solve(A, b, pivot=pivot)

        display_latex(b,             label=r"b")
        display_latex(x_hat,         label=r"\hat x")
        display(Markdown(fr"режим выбора главного элемента: `{pivot}`"))
        display_latex(info["U"],     label=r"U")
        display_latex(info["b_mod"], label=r"\tilde b")
        det_str = f"{info['det']:.16g}"
        display(Markdown(rf"$\det(A)$ (по диагонали U) $= {det_str}$"))

        A_arr = _as_array_M(A); b_arr = _as_array_v(b); x_arr = x_hat.data
        r = A_arr @ x_arr - b_arr
        rel_res = np.linalg.norm(r, 2) / (np.linalg.norm(A_arr, 2)*np.linalg.norm(x_arr, 2))
        display_latex(rel_res, label=r"\mathrm{rel\_res}")

        if x_true is not None:
            x_true_arr = _as_array_v(x_true)
            err = x_arr - x_true_arr
            rel_err = np.linalg.norm(err, 2) / np.linalg.norm(x_true_arr, 2)
            display_latex(rel_err, label=r"\mathrm{rel\_err}")
            abs_err2 = float(np.linalg.norm(err, 2))
            display_latex(abs_err2, label=r"\|\hat x - x\|_2")

        th = gauss_ops_theory(A_arr.shape[0])
        display_latex(info["ops"]["total"], label=r"Q_{\mathrm{meas}}")
        display_latex(th["total"],           label=r"Q_{\mathrm{theory}}")

        print("мяу, готово (=^..^=)")
        return

    if mode == "3":
        log_every = input("логировать каждый k-шаг? (число, пусто = нет): ").strip()
        log_every = int(log_every) if log_every else None

        try:
            L_lu, U_lu, _ = lu_nopivot(A, log_every=log_every)
            display(Markdown("**LU-разложение (без перестановок):**"))
            display_latex(L_lu, label=r"L")
            display_latex(U_lu, label=r"U")
            display(Markdown(rf"$\det(A)$ из LU $= {det_from_U(U_lu):.16g}$"))
        except Exception as e:
            display(Markdown(f"**LU без перестановок не удалось:** `{e}`"))
            print("мяу, готово (=^..^=)")
            return

        if input("решить сейчас Ax=b через LU? (y/n): ").strip().lower() == "y":
            if b is None:
                line = input("введите b (n чисел через пробел): ").strip()
                b = Vec(np.array([float(x) for x in line.replace(',', ' ').split()], dtype=float))
            x_lu, _ = solve_via_LU(L_lu, U_lu, b)
            display_latex(b,    label=r"b")
            display_latex(x_lu, label=r"x_{\mathrm{LU}}")
            A_arr = _as_array_M(A); x_arr = x_lu.data; b_arr = _as_array_v(b)
            r = A_arr @ x_arr - b_arr
            rel_res = np.linalg.norm(r, 2) / (np.linalg.norm(A_arr, 2)*np.linalg.norm(x_arr, 2))
            display_latex(rel_res, label=r"\mathrm{rel\_res}")
            # если есть x_true — покажем и ошибки
            if x_true is not None:
                err = x_arr - _as_array_v(x_true)
                rel_err = np.linalg.norm(err, 2) / np.linalg.norm(_as_array_v(x_true), 2)
                display_latex(rel_err, label=r"\mathrm{rel\_err}")
                display_latex(float(np.linalg.norm(err,2)), label=r"\|\hat x - x\|_2")
        print("мяу, готово (=^..^=)")
        return

    if mode == "4":
        log_every = input("логировать каждый k-шаг? (число, пусто = нет): ").strip()
        log_every = int(log_every) if log_every else None

        try:
            P, Lp, Up, swaps, _ = lup(A, log_every=log_every)
            display(Markdown("**LUP-разложение (частичный pivot):**"))
            display_latex(P,  label=r"P")
            display_latex(Lp, label=r"L")
            display_latex(Up, label=r"U")
            display(Markdown(rf"$\det(A)$ из LUP $= {det_from_U(Up, swaps):.16g}$"))
        except Exception as e:
            display(Markdown(f"**LUP не удалось:** `{e}`"))
            print("мяу, готово (=^..^=)")
            return

        if input("решить сейчас Ax=b через LUP? (y/n): ").strip().lower() == "y":
            if b is None:
                line = input("введите b (n чисел через пробел): ").strip()
                b = Vec(np.array([float(x) for x in line.replace(',', ' ').split()], dtype=float))
            x_lup, _ = solve_via_LUP(P, Lp, Up, b)
            display_latex(b,     label=r"b")
            display_latex(x_lup, label=r"x_{\mathrm{LUP}}")
            A_arr = _as_array_M(A); x_arr = x_lup.data; b_arr = _as_array_v(b)
            r = A_arr @ x_arr - b_arr
            rel_res = np.linalg.norm(r, 2) / (np.linalg.norm(A_arr, 2)*np.linalg.norm(x_arr, 2))
            display_latex(rel_res, label=r"\mathrm{rel\_res}")
            if x_true is not None:
                err = x_arr - _as_array_v(x_true)
                rel_err = np.linalg.norm(err, 2) / np.linalg.norm(_as_array_v(x_true), 2)
                display_latex(rel_err, label=r"\mathrm{rel\_err}")
                display_latex(float(np.linalg.norm(err,2)), label=r"\|\hat x - x\|_2")
        print("мяу, готово (=^..^=)")
        return

    if mode == "5":
        pivot = _parse_pivot(input("pivot для Гаусса? (`none`/`col`/`row`/`full`, по умолчанию col): "))
        try:
            m = int(input("сколько правых частей m = ").strip() or "5")
        except Exception:
            m = 5
        run_many_rhs_uniform(A, m, pivot=pivot, a=-10.0, b=10.0, seed=42)
        display(Markdown("---"))
        compare_ops(A, m, pivot_for_gauss=pivot, a=-10.0, b=10.0, seed=42)
        print("мяу, готово (=^..^=)")
        return

main()