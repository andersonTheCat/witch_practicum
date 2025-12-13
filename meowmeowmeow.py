import numpy as np
from IPython.display import display, Markdown

from meowmeow import (
    Mat, Vec, display_latex, _err,
    _as_array_M, _as_array_v,
    gauss_solve, gauss_ops_theory, pre_assess,
    _ops_zero, _ops_inc, _ops_merge,
)
def _bullet(ok: bool):
    return "уютно: " if ok else "неуютно: "
def _vnorm(x: np.ndarray, kind: str = "2"):
    k = kind.lower()
    if k in {"2", "euclid"}:
        return float(np.linalg.norm(x, 2))
    if k in {"1"}:
        return float(np.linalg.norm(x, 1))
    if k in {"inf", "infty"}:
        return float(np.linalg.norm(x, np.inf))
    raise _err("норма вектора поддерживает только kind из {'1','2','inf'}")
def _mnorm(A: np.ndarray, kind: str = "inf"):
    k = kind.lower()
    if k == "1":
        return float(np.linalg.norm(A, 1))
    if k in {"inf", "infty"}:
        return float(np.linalg.norm(A, np.inf))
    if k == "2":
        return float(np.linalg.norm(A, 2))
    raise _err("норма матрицы поддерживает только kind из {'1','2','inf'}")
def _diag_dominance_report(A: np.ndarray):
    n = A.shape[0]
    diag = np.abs(np.diag(A))
    off = np.sum(np.abs(A), axis=1) - diag
    strict = np.all(diag > off)
    weak = np.all(diag >= off)
    margin = float(np.min(diag - off))
    return {"weak": bool(weak), "strict": bool(strict), "margin": margin, "diag": diag, "off": off}
def _print_system_stats(A: Mat, b: Vec, B: Mat | None = None):
    Aarr = _as_array_M(A)
    stats = pre_assess(A)

    display(Markdown("**система $Ax=b$**"))
    display_latex(A, label=r"A")
    display_latex(b, label=r"b")
    display(Markdown("всякое (чтобы понять, насколько система добренькая):"))
    display_latex(stats["cond2"], label=r"\mathrm{cond}_2(A)")
    display_latex(stats["det"],   label=r"\det(A)")

    dd = _diag_dominance_report(Aarr)
    display(Markdown("диагональное преобладание (строчное):"))

    tex_weak   = r"$|a_{ii}|\ge \sum_{j\ne i}|a_{ij}|$"
    tex_strict = r"$|a_{ii}|> \sum_{j\ne i}|a_{ij}|$"
    tex_margin = r"$\min_i\left(|a_{ii}|-\sum_{j\ne i}|a_{ij}|\right)$"

    display(Markdown(f"{_bullet(dd['weak'])}  {tex_weak}"))
    display(Markdown(f"{_bullet(dd['strict'])}  {tex_strict}"))
    display(Markdown(f"минимальный запас: {tex_margin} = `{dd['margin']:.6g}`"))

    if B is not None:
        Barr = _as_array_M(B)
        display(Markdown("**матрица итерации $B$ (Якоби):**"))
        display_latex(B, label=r"B")
        n1  = _mnorm(Barr, "1")
        ni  = _mnorm(Barr, "inf")
        try:
            rho = float(np.max(np.abs(np.linalg.eigvals(Barr))))
        except Exception:
            rho = float("nan")
        display_latex(n1, label=r"\|B\|_1")
        display_latex(ni, label=r"\|B\|_\infty")
        display_latex(rho, label=r"\rho(B)")
def jacobi_prepare(A_in, b_in):
    A = _as_array_M(A_in)
    b = _as_array_v(b_in)
    n, m = A.shape
    if n != m:
        raise _err("якоби кушает только квадратную матрицу A")
    if b.shape != (n,):
        raise _err("размерность b должна быть (n,)")

    diag = np.diag(A).copy()
    if np.any(diag == 0.0):
        raise _err("на диагонали A есть нули")

    ops = _ops_zero()

    B = np.zeros_like(A, dtype=float)
    c = np.zeros(n, dtype=float)

    # b_ij = -a_ij/a_ii, c_i = b_i/a_ii; b_ii = 0
    for i in range(n):
        c[i] = b[i] / diag[i]; _ops_inc(ops, "div")
        for j in range(n):
            if i == j:
                B[i, j] = 0.0
            else:
                B[i, j] = -(A[i, j] / diag[i]); _ops_inc(ops, "div")

    ops["total"] = sum(ops[k] for k in ops if k != "total")
    info = {"ops_prepare": ops}
    return Mat(B), Vec(c), info
def make_x0(kind: str, n: int, b: Vec, c: Vec, seed: int = 42):
    k = (kind or "").strip().lower()
    if k in {"0", "zero", "zeros", "н", "нули", "нуль"}:
        return Vec(np.zeros(n))
    if k in {"1", "one", "ones", "е", "ед", "единицы"}:
        return Vec(np.ones(n))
    if k in {"b", "rhs", "п", "правая", "праваячасть"}:
        return Vec(b.data.copy())
    if k in {"c", "const", "с", "конст", "константа"}:
        return Vec(c.data.copy())
    if k in {"rand", "random", "сл", "случ", "случайный"}:
        rng = np.random.default_rng(seed)
        return Vec(rng.uniform(-1.0, 1.0, size=n))
    raise _err("x0 kind должен быть одним из: zeros/ones/b/c/rand")
def _Bx_plus_c_with_ops(B: np.ndarray, x: np.ndarray, c: np.ndarray, ops):
    n = B.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s = s + B[i, j] * x[j]
            _ops_inc(ops, "mul")
            _ops_inc(ops, "add")
        out[i] = s + c[i]
        _ops_inc(ops, "add")
    return out
def _stop_check(stop_rule: str,
                A: np.ndarray, b: np.ndarray,
                Bnorm: float,
                x_new: np.ndarray, x_old: np.ndarray,
                eps: float, vec_norm_kind: str):
    delta = _vnorm(x_new - x_old, vec_norm_kind)

    if stop_rule == "delta":
        return (delta < eps), {"delta": delta}

    if stop_rule == "delta_scaled":
        if Bnorm <= 0 or Bnorm >= 1:
            return False, {"delta": delta, "eps1": float("nan")}
        eps1 = (1.0 - Bnorm) / Bnorm * eps
        return (delta < eps1), {"delta": delta, "eps1": eps1}

    if stop_rule == "aposteriori":
        if Bnorm <= 0 or Bnorm >= 1:
            return False, {"delta": delta, "bound": float("inf")}
        bound = (Bnorm / (1.0 - Bnorm)) * delta
        return (bound < eps), {"delta": delta, "bound": bound}

    if stop_rule == "residual":
        r = A @ x_new - b
        rn = _vnorm(r, vec_norm_kind)
        return (rn < eps), {"delta": delta, "res": rn}

    raise _err("stop_rule должен быть одним из: delta / delta_scaled / aposteriori / residual")
def jacobi_solve(A_in, b_in, x0: Vec,
                eps: float = 1e-6,
                max_iter: int = 500,
                stop_rule: str = "aposteriori",
                Bnorm_kind: str = "inf",
                vec_norm_kind: str = "2",
                log_each: bool = True):
    A = _as_array_M(A_in)
    b = _as_array_v(b_in)
    n = A.shape[0]

    B, c, prep = jacobi_prepare(Mat(A), Vec(b))
    Barr = _as_array_M(B)
    carr = _as_array_v(c)

    try:
        Bnorm = _mnorm(Barr, Bnorm_kind)
    except Exception:
        Bnorm = float("nan")

    ops_iter = _ops_zero()
    ops_total = _ops_merge(prep["ops_prepare"], ops_iter)

    x_old = _as_array_v(x0).copy()
    if x_old.shape != (n,):
        raise _err("x0 должен иметь размерность (n,)")

    history = []
    if log_each:
        display(Markdown(
            f"Якоби: стоп=`{stop_rule}`, "
            f"$\\|B\\|_{{{Bnorm_kind}}}$=`{Bnorm:.6g}`, "
            f"норма вектора=`{vec_norm_kind}`, eps=`{eps}`"
        ))
        display_latex(Vec(x_old), label=r"x^{(0)}")

    prev_delta = None
    for k in range(0, max_iter):
        x_new = _Bx_plus_c_with_ops(Barr, x_old, carr, ops_iter)
        ok, extra = _stop_check(stop_rule, A, b, Bnorm, x_new, x_old, eps, vec_norm_kind)
        delta = extra.get("delta", float("nan"))

        q_est = None
        if prev_delta is not None and prev_delta > 0:
            q_est = delta / prev_delta

        row = {
            "k": k+1,
            "x": x_new.copy(),
            "delta": float(delta),
            "q_est": (None if q_est is None else float(q_est)),
            **extra
        }
        history.append(row)

        if log_each:
            msg = (
                f"**шаг {k+1}:**  "
                f"$\\|x^{{({k+1})}}-x^{{({k})}}\\|$ = `{delta:.6g}`"
            )
            if q_est is not None:
                msg += f",  q_est ≈ `{q_est:.6g}`"
            if "bound" in extra:
                msg += f",  апост.оценка = `{extra['bound']:.6g}`"
            if "eps1" in extra:
                msg += f",  eps1 = `{extra['eps1']:.6g}`"
            if "res" in extra:
                msg += f",  ||Ax-b|| = `{extra['res']:.6g}`"
            display(Markdown(msg))
            display_latex(Vec(x_new), label=rf"x^{{({k+1})}}")

        if ok:
            x_hat = Vec(x_new)
            ops_total = _ops_merge(prep["ops_prepare"], ops_iter)
            return x_hat, {
                "B": B, "c": c, "Bnorm": Bnorm,
                "ops_prepare": prep["ops_prepare"],
                "ops_iter": ops_iter,
                "ops_total": ops_total,
                "iters": k+1,
                "history": history,
                "stop_rule": stop_rule,
                "Bnorm_kind": Bnorm_kind,
                "vec_norm_kind": vec_norm_kind
            }

        prev_delta = delta
        x_old = x_new

    x_hat = Vec(x_old)
    ops_total = _ops_merge(prep["ops_prepare"], ops_iter)
    return x_hat, {
        "B": B, "c": c, "Bnorm": Bnorm,
        "ops_prepare": prep["ops_prepare"],
        "ops_iter": ops_iter,
        "ops_total": ops_total,
        "iters": max_iter,
        "history": history,
        "stop_rule": stop_rule,
        "Bnorm_kind": Bnorm_kind,
        "vec_norm_kind": vec_norm_kind,
        "warn": "достигнут max_iter, но критерий не сработал"
    }
def _post_metrics(A: Mat, b: Vec, x_hat: Vec, x_true: Vec | None = None, vec_norm_kind: str = "2"):
    Aarr = _as_array_M(A)
    barr = _as_array_v(b)
    xarr = _as_array_v(x_hat)
    r = Aarr @ xarr - barr

    rn = _vnorm(r, vec_norm_kind)
    display(Markdown("**уютность:**"))
    display_latex(rn, label=rf"\|Ax-b\|_{{{vec_norm_kind}}}")

    if x_true is not None:
        xt = _as_array_v(x_true)
        e = xarr - xt
        en = _vnorm(e, vec_norm_kind)
        rel = en / (_vnorm(xt, vec_norm_kind) + 1e-300)
        display_latex(en,  label=rf"\|\hat x-x\|_{{{vec_norm_kind}}}")
        display_latex(rel, label=r"\mathrm{rel\_err}")
def compare_stopping_rules(A: Mat, b: Vec, x0: Vec, eps: float,
                           Bnorm_kind="inf", vec_norm_kind="2", max_iter=500):
    rules = ["aposteriori", "delta_scaled", "delta", "residual"]
    display(Markdown(f"\n\n**сравнение критериев остановки (eps={eps})**\n\n"))
    rows = []
    for rule in rules:
        xh, info = jacobi_solve(A, b, x0, eps=eps, max_iter=max_iter,
                                stop_rule=rule, Bnorm_kind=Bnorm_kind,
                                vec_norm_kind=vec_norm_kind, log_each=False)
        rows.append((rule, info["iters"], info["ops_total"]["total"], info.get("warn")))
    for rule, iters, q, warn in rows:
        w = "" if warn is None else f"  мяк: `{warn}`"
        display(Markdown(f"`{rule}`: итераций = **{iters}**,  операций = **{q}**; {w}"))
def compare_with_gauss(A: Mat, b: Vec, x_hat: Vec, ops_jacobi_total: dict, pivot="col"):
    display(Markdown("**сравнение с Гауссом**"))
    xg, info = gauss_solve(A, b, pivot=pivot)
    display_latex(xg, label=r"x_{\mathrm{gauss}}")

    Aarr = _as_array_M(A); barr = _as_array_v(b)
    rg = Aarr @ xg.data - barr
    rn = float(np.linalg.norm(rg, 2))
    display_latex(rn, label=r"\|A x_{\mathrm{gauss}}-b\|_2")

    display(Markdown("**операции:**"))
    display(Markdown(f"Якоби (всего): `Q = {ops_jacobi_total['total']}`"))
    display(Markdown(f"Гаусс:          `Q = {info['ops']['total']}`"))
    th = gauss_ops_theory(Aarr.shape[0])
    display(Markdown(f"Гаусс (теория, одна система): `Q_theory ≈ {th['total']:.6g}`"))
def make_dd_system(n: int, seed: int = 42, strength: float = 2.0, x_range=(-5.0, 5.0)):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + strength
    x_true = rng.uniform(x_range[0], x_range[1], size=n)
    b = A @ x_true
    return Mat(A), Vec(b), Vec(x_true)
def make_borderline_system(n: int, seed: int = 43, slack: float = 1e-2, x_range=(-5.0, 5.0)):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(n, n))
    for i in range(n):
        off = np.sum(np.abs(A[i, :])) - abs(A[i, i])
        A[i, i] = off + slack
    x_true = rng.uniform(x_range[0], x_range[1], size=n)
    b = A @ x_true
    return Mat(A), Vec(b), Vec(x_true)
def make_bad_system(n: int, seed: int = 44):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(n, n))
    for i in range(n):
        A[i, i] = rng.uniform(-0.2, 0.2)
        if abs(A[i, i]) < 1e-2:
            A[i, i] = 0.05
    x_true = rng.uniform(-2.0, 2.0, size=n)
    b = A @ x_true
    return Mat(A), Vec(b), Vec(x_true)
def solve_for_epsilons(A: Mat, b: Vec, x0: Vec,
                       eps_list=(1e-3, 1e-6),
                       stop_rule="aposteriori",
                       Bnorm_kind="inf", vec_norm_kind="2",
                       max_iter=500, log_each=True,
                       x_true: Vec | None = None,
                       want_gauss=True):
    B, c, _ = jacobi_prepare(A, b)
    _print_system_stats(A, b, B=B)

    display(Markdown("**запуск итераций**"))
    for eps in eps_list:
        display(Markdown(f"\nточность eps = `{eps}`"))
        x_hat, info = jacobi_solve(A, b, x0, eps=eps, max_iter=max_iter,
                                   stop_rule=stop_rule, Bnorm_kind=Bnorm_kind,
                                   vec_norm_kind=vec_norm_kind, log_each=log_each)
        display(Markdown(f"**итог:** итераций = **{info['iters']}**"))
        display_latex(x_hat, label=r"\hat x")
        _post_metrics(A, b, x_hat, x_true=x_true, vec_norm_kind=vec_norm_kind)

        display(Markdown("**операции:**"))
        display(Markdown(f"подготовка (B,c): `Q = {info['ops_prepare']['total']}`"))
        display(Markdown(f"итерации:         `Q = {info['ops_iter']['total']}`"))
        display(Markdown(f"всего:            `Q = {info['ops_total']['total']}`"))

        n = A.shape[0]
        q_iter_theory = 2*n*n + n
        display(Markdown(f"в теории: `Q_iter ≈ 2 n^2 + n = {q_iter_theory}` за итерацию (плюс подготовка)"))

        if want_gauss:
            compare_with_gauss(A, b, x_hat, info["ops_total"], pivot="col")

        compare_stopping_rules(A, b, x0, eps, Bnorm_kind=Bnorm_kind, vec_norm_kind=vec_norm_kind, max_iter=max_iter)

    print("мяу, готово (=^..^=)")
def demo_three_inputs(n=5):
    display(Markdown("**1) входные данные №1: добренькая диагонально преобладающая**"))
    A1, b1, x1 = make_dd_system(n, seed=7, strength=2.5)
    B1, c1, _ = jacobi_prepare(A1, b1)
    x0_1 = make_x0("zeros", n, b1, c1)
    solve_for_epsilons(A1, b1, x0_1, x_true=x1, log_each=True)

    display(Markdown("\n\n**2) входные данные №2: почти на грани (обычно очень медленно)**"))
    A2, b2, x2 = make_borderline_system(n, seed=8, slack=1e-2)
    B2, c2, _ = jacobi_prepare(A2, b2)
    x0_2 = make_x0("zeros", n, b2, c2)
    solve_for_epsilons(A2, b2, x0_2, x_true=x2, log_each=False)

    display(Markdown("\n\n**3) входные данные №3: плохонькая (может не сходиться)**"))
    A3, b3, x3 = make_bad_system(n, seed=9)
    B3, c3, _ = jacobi_prepare(A3, b3)
    x0_3 = make_x0("zeros", n, b3, c3)
    solve_for_epsilons(A3, b3, x0_3, x_true=x3, log_each=False, max_iter=50, want_gauss=False)
def main():
    display(Markdown(
        "мяу мяу, что выбираем?\n\n"
        "[1] демка на трёх входных данных (уютно / медленно / неуютно)\n\n"
        "[2] решить свою систему (ручной ввод)\n\n"
        "[3] сгенерировать диагонально преобладающую систему и решить\n"
    ))
    mode = (input("выбор (1/2/3): ").strip() or "1")

    if mode == "1":
        n = int(input("n (по умолчанию 5): ").strip() or "5")
        demo_three_inputs(n=n)
        return

    if mode == "2":
        from meow import _read_matrix, _read_vector
        display(Markdown("**ввод A:**"))
        A = _read_matrix()
        if not A.is_square():
            raise _err("нужно ввести квадратную A")
        display(Markdown("**ввод b:**"))
        b = _read_vector("b (n чисел): ")

        B, c, _ = jacobi_prepare(A, b)
        n = A.shape[0]

        display(Markdown("теперь нужно выбрать x0: `zeros` / `ones` / `b` / `c` / `rand`"))
        x0_kind = input("x0 kind → ").strip() or "zeros"
        x0 = make_x0(x0_kind, n, b, c)

        display(Markdown(
            "**критерии остановки:**  \n"
            "`delta`: остановка по шагу  $\\|x^{(k)}-x^{(k-1)}\\| < \\varepsilon$  \n"
            "`delta_scaled`: шаг с поправкой  $\\|x^{(k)}-x^{(k-1)}\\| < \\frac{1-\\|B\\|}{\\|B\\|}\\,\\varepsilon$ (нужно $\\|B\\|<1$)  \n"
            "`aposteriori`: апост. оценка ошибки  $\\frac{\\|B\\|}{1-\\|B\\|}\\,\\|x^{(k)}-x^{(k-1)}\\| < \\varepsilon$ (нужно $\\|B\\|<1$)  \n"
            "`residual`: по невязке  $\\|Ax^{(k)}-b\\| < \\varepsilon$  \n"
        ))
        stop_rule = input("stop (aposteriori/delta_scaled/delta/residual) → ").strip() or "aposteriori"

        eps1 = float(input("eps для 1-го прогона (по умолчанию 1e-3): ").strip() or "1e-3")
        eps2 = float(input("eps для 2-го прогона (по умолчанию 1e-6): ").strip() or "1e-6")

        log_each = (input("печатать каждую итерацию? (y/n, по умолчанию y): ").strip().lower() or "y").startswith("y")
        solve_for_epsilons(A, b, x0, eps_list=(eps1, eps2),
                           stop_rule=stop_rule, Bnorm_kind="inf",
                           vec_norm_kind="2", max_iter=500,
                           log_each=log_each, x_true=None, want_gauss=True)
        return

    if mode == "3":
        n = int(input("n (по умолчанию 6): ").strip() or "6")
        seed = int(input("seed (по умолчанию 42): ").strip() or "42")
        strength = float(input("запас диагонального преобладания δ (|a_ii| − ∑_{j≠i}|a_ij|), по умолчанию 2.0: ").strip() or "2.0")
        A, b, x_true = make_dd_system(n, seed=seed, strength=strength)

        B, c, _ = jacobi_prepare(A, b)
        display(Markdown("выбераем x0: `zeros` / `ones` / `b` / `c` / `rand`"))
        x0_kind = input("x0 kind → ").strip() or "zeros"
        x0 = make_x0(x0_kind, n, b, c, seed=seed+1)

        display(Markdown(
            "**критерии остановки:**  \n"
            "`delta`: остановка по шагу  $\\|x^{(k)}-x^{(k-1)}\\| < \\varepsilon$  \n"
            "`delta_scaled`: шаг с поправкой  $\\|x^{(k)}-x^{(k-1)}\\| < \\frac{1-\\|B\\|}{\\|B\\|}\\,\\varepsilon$ (нужно $\\|B\\|<1$)  \n"
            "`aposteriori`: апост. оценка ошибки  $\\frac{\\|B\\|}{1-\\|B\\|}\\,\\|x^{(k)}-x^{(k-1)}\\| < \\varepsilon$ (нужно $\\|B\\|<1$)  \n"
            "`residual`: по невязке  $\\|Ax^{(k)}-b\\| < \\varepsilon$  \n"
        ))
        stop_rule = input("stop (aposteriori/delta_scaled/delta/residual) [aposteriori]: ").strip() or "aposteriori"
        log_each = (input("печатать каждую итерацию? (y/n, по умолчанию n): ").strip().lower() or "n").startswith("y")

        solve_for_epsilons(A, b, x0, eps_list=(1e-3, 1e-6),
                           stop_rule=stop_rule, Bnorm_kind="inf",
                           vec_norm_kind="2", max_iter=500,
                           log_each=log_each, x_true=x_true, want_gauss=True)
        return

    print("мяу мяу, выбери 1/2/3")

if __name__ == "__main__":
    main()