import numpy as np
from IPython.display import display, Markdown

from meowmeow import (
    Mat, Vec, display_latex, _err,
    _as_array_M, _as_array_v,
    _ops_zero, _ops_inc, _ops_merge,
)
from meowmeowmeow import (
    _vnorm, _mnorm,
)
def _bullet(ok: bool):
    return "уютно: " if ok else "неуютно: "
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")
def _dot_with_ops(x: np.ndarray, y: np.ndarray, ops):
    s = 0.0
    n = x.shape[0]
    for i in range(n):
        s = s + x[i] * y[i]
        _ops_inc(ops, "mul")
        _ops_inc(ops, "add")
    return s
def _matvec_with_ops(A: np.ndarray, x: np.ndarray, ops):
    n, m = A.shape
    if x.shape != (m,):
        raise _err("matvec: размерности не совпали")
    y = np.zeros(n, dtype=float)
    for i in range(n):
        s = 0.0
        for j in range(m):
            s = s + A[i, j] * x[j]
            _ops_inc(ops, "mul")
            _ops_inc(ops, "add")
        y[i] = s
    return y
def rayleigh_quotient(A: np.ndarray, x: np.ndarray, ops=None):
    if ops is None:
        ops = _ops_zero()
    Ax = _matvec_with_ops(A, x, ops)
    num = _dot_with_ops(Ax, x, ops)
    den = _dot_with_ops(x, x, ops)
    if den == 0.0:
        raise _err("в отношении Рэлея (x,x)=0 - неуютно")
    _ops_inc(ops, "div")
    return num / den
def gershgorin_disks(A: np.ndarray):
    n = A.shape[0]
    centers = np.diag(A).copy()
    radii = np.zeros(n, dtype=float)
    for i in range(n):
        radii[i] = float(np.sum(np.abs(A[i, :]))) - abs(float(A[i, i]))
    segL = centers - radii
    segR = centers + radii
    return centers, radii, segL, segR
def _print_task_stats(A: Mat, show_true_eigs: bool = True):
    Aarr = _as_array_M(A)
    display(Markdown("**матрица $A$**"))
    display_latex(A, label=r"A")

    try:
        c2 = float(np.linalg.cond(Aarr, 2))
    except Exception:
        c2 = float("nan")
    display(Markdown(f"**индикаторы:**  \n`cond2(A)` = `{c2:.6g}`"))

    centers, radii, segL, segR = gershgorin_disks(Aarr)
    display(Markdown("**круги Гершгорина**  \n(для вещественного случая ещё и отрезки $[a_{ii}-r_i, a_{ii}+r_i]$):"))
    for i in range(Aarr.shape[0]):
        display(Markdown(
            f"i={i+1}:  центр `a_ii={centers[i]:.6g}`,  радиус `r_i={radii[i]:.6g}`,  "
            f"отрезок `[ {segL[i]:.6g}, {segR[i]:.6g} ]`"
        ))

    if show_true_eigs:
        try:
            eigs = np.linalg.eigvals(Aarr)
            j = int(np.argmax(np.abs(eigs)))
            lam_star = eigs[j]
            display(Markdown("**проверка (numpy)**"))
            display(Markdown(f"max по модулю: `λ_maxabs ≈ {lam_star}`  (|λ|≈{abs(lam_star):.6g})"))
        except Exception:
            display(Markdown("**проверка (numpy):** не удалось посчитать eigvals :<"))
def make_x0_eig(kind: str, n: int, seed: int = 42):
    k = (kind or "").strip().lower()
    if k in {"ones", "1", "ед", "единицы"}:
        return Vec(np.ones(n))
    if k in {"rand", "random", "сл", "случ"}:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1.0, 1.0, size=n)
        if _vnorm(x, "2") == 0.0:
            x[0] = 1.0
        return Vec(x)
    if k.startswith("e"):
        try:
            idx = int(k[1:]) - 1
            if not (0 <= idx < n):
                raise ValueError()
            x = np.zeros(n); x[idx] = 1.0
            return Vec(x)
        except Exception:
            raise _err("x0 kind вида e1/e2/.../en")
    raise _err("x0 kind: ones / rand / e1..en")
def power_plain(A_in, x0: Vec,
                eps: float = 1e-6,
                max_iter: int = 500,
                stop_rule: str = "residual",
                vec_norm_kind: str = "2",
                log_each: bool = True):
    A = _as_array_M(A_in)
    n, m = A.shape
    if n != m:
        raise _err("степенной метод просит квадратную A")

    ops_iter = _ops_zero()

    # вот фикс: приводим тут x0 к вектору длины n
    x_old = np.asarray(_as_array_v(x0), dtype=float).reshape(-1).copy()
    if x_old.shape != (n,):
        raise _err("x0 должен иметь размерность (n,)")

    if _vnorm(x_old, "2") == 0.0:
        raise _err("x0 не должен быть нулевым")

    lam_old = None
    history = []

    if log_each:
        display(Markdown(
            f"**степенной метод (обычный):** stop=`{stop_rule}`, eps=`{eps}`, max_iter=`{max_iter}`"
        ))
        display_latex(Vec(x_old), label=r"x^{(0)}")

    for k in range(1, max_iter + 1):
        x_new = _matvec_with_ops(A, x_old, ops_iter)

        num = _dot_with_ops(x_new, x_old, ops_iter)
        den = _dot_with_ops(x_old, x_old, ops_iter)
        if den == 0.0:
            raise _err("на шаге получился нулевой (x_old,x_old)")
        _ops_inc(ops_iter, "div")
        lam = num / den

        xn = _vnorm(x_new, "2")
        x_hat = x_new.copy() if xn == 0.0 else (x_new / xn)

        Ax_hat = A @ x_hat
        r = Ax_hat - lam * x_hat
        sigma = _vnorm(r, vec_norm_kind) / (_vnorm(x_hat, vec_norm_kind) + 1e-300)

        ok = False
        extra = {"sigma": float(sigma)}
        if stop_rule == "residual":
            ok = (sigma < eps)
        elif stop_rule == "lambda":
            if lam_old is not None:
                extra["dlam"] = float(abs(lam - lam_old))
                ok = (abs(lam - lam_old) < eps)
        else:
            raise _err("stop_rule: residual / lambda")

        history.append({"k": k, "lambda": float(lam), "sigma": float(sigma), **extra})

        if log_each:
            msg = f"**шаг {k}:**  λ ≈ `{lam:.12g}`,  sigma=||Ax-λx||/||x|| ≈ `{sigma:.6g}`"
            if "dlam" in extra:
                msg += f",  |Δλ| ≈ `{extra['dlam']:.6g}`"
            display(Markdown(msg))
            display_latex(Vec(x_hat), label=rf"\hat e_1^{({k})}")

        if ok:
            e1 = Vec(x_hat)
            info = {
                "iters": k,
                "lambda": float(lam),
                "sigma": float(sigma),
                "ops_iter": ops_iter,
                "ops_total": ops_iter,
                "history": history,
                "method": "plain",
                "stop_rule": stop_rule
            }
            return e1, info

        lam_old = lam
        x_old = x_new

    xn = _vnorm(x_old, "2")
    e1 = Vec(x_old if xn == 0.0 else (x_old / xn))
    info = {
        "iters": max_iter,
        "lambda": (float(lam_old) if lam_old is not None else float("nan")),
        "sigma": float("nan"),
        "ops_iter": ops_iter,
        "ops_total": ops_iter,
        "history": history,
        "method": "plain",
        "stop_rule": stop_rule,
        "warn": "достигнут max_iter, но критерий не сработал"
    }
    return e1, info
def power_normalized(A_in, x0: Vec,
                     eps: float = 1e-6,
                     max_iter: int = 500,
                     stop_rule: str = "residual",
                     vec_norm_kind: str = "2",
                     log_each: bool = True):
    A = _as_array_M(A_in)
    n, m = A.shape
    if n != m:
        raise _err("степенной метод просит квадратную A")

    ops_iter = _ops_zero()

    # и вот фикс - тоже приводим тут x0 к вектору длины n (=^..^=)
    x_old = np.asarray(_as_array_v(x0), dtype=float).reshape(-1).copy()
    if x_old.shape != (n,):
        raise _err("x0 должен иметь размерность (n,)")

    nx = _vnorm(x_old, "2")
    if nx == 0.0:
        raise _err("x0 не должен быть нулевым")
    x_old = x_old / nx

    lam_old = None
    history = []

    if log_each:
        display(Markdown(
            f"**степенной метод (с нормировкой):** stop=`{stop_rule}`, eps=`{eps}`, max_iter=`{max_iter}`"
        ))
        display_latex(Vec(x_old), label=r"x^{(0)} \;(\|x^{(0)}\|_2=1)")

    for k in range(1, max_iter + 1):
        y = _matvec_with_ops(A, x_old, ops_iter)
        lam = _dot_with_ops(y, x_old, ops_iter)

        ny = _vnorm(y, "2")
        if ny == 0.0:
            raise _err("получилось y=0, нормировка невозможна")
        x_new = y / ny

        r = (A @ x_new) - lam * x_new
        sigma = _vnorm(r, vec_norm_kind) / (_vnorm(x_new, vec_norm_kind) + 1e-300)

        ok = False
        extra = {"sigma": float(sigma)}
        if stop_rule == "residual":
            ok = (sigma < eps)
        elif stop_rule == "lambda":
            if lam_old is not None:
                extra["dlam"] = float(abs(lam - lam_old))
                ok = (abs(lam - lam_old) < eps)
        else:
            raise _err("stop_rule: residual / lambda")

        history.append({"k": k, "lambda": float(lam), "sigma": float(sigma), **extra})

        if log_each:
            msg = f"**шаг {k}:**  λ ≈ `{lam:.12g}`,  sigma ≈ `{sigma:.6g}`"
            if "dlam" in extra:
                msg += f",  |Δλ| ≈ `{extra['dlam']:.6g}`"
            display(Markdown(msg))
            display_latex(Vec(x_new), label=rf"\hat e_1^{({k})}")

        if ok:
            e1 = Vec(x_new)
            info = {
                "iters": k,
                "lambda": float(lam),
                "sigma": float(sigma),
                "ops_iter": ops_iter,
                "ops_total": ops_iter,
                "history": history,
                "method": "normalized",
                "stop_rule": stop_rule
            }
            return e1, info

        lam_old = lam
        x_old = x_new

    e1 = Vec(x_old)
    info = {
        "iters": max_iter,
        "lambda": (float(lam_old) if lam_old is not None else float("nan")),
        "sigma": float("nan"),
        "ops_iter": ops_iter,
        "ops_total": ops_iter,
        "history": history,
        "method": "normalized",
        "stop_rule": stop_rule,
        "warn": "достигнут max_iter, но критерий не сработал"
    }
    return e1, info
def _compare_with_numpy(A: Mat, lam_hat: float, e_hat: Vec):
    Aarr = _as_array_M(A)
    x = _as_array_v(e_hat)
    try:
        eigs = np.linalg.eigvals(Aarr)
        j = int(np.argmax(np.abs(eigs)))
        lam_true = eigs[j]
        display(Markdown("**сравнение с numpy.eigvals**"))
        display(Markdown(f"λ_hat ≈ `{lam_hat}`"))
        display(Markdown(f"λ_true(max|·|) ≈ `{lam_true}`   (|λ_true|≈{abs(lam_true):.6g})"))
        r = Aarr @ x - lam_hat * x
        display(Markdown(f"невязка для (λ_hat, e_hat): `||Ax-λx||_2` ≈ `{np.linalg.norm(r):.6g}`"))
    except Exception:
        display(Markdown("**numpy сравнение:** не удалось (("))
def solve_for_epsilons_power(A: Mat, x0: Vec,
                             eps_list=(1e-3, 1e-6),
                             stop_rule="residual",
                             max_iter=500,
                             log_each=True,
                             show_true_eigs=True):
    _print_task_stats(A, show_true_eigs=show_true_eigs)

    display(Markdown("**запуски для eps**"))
    for eps in eps_list:
        display(Markdown(f"**точность eps = `{eps}`**"))

        e_plain, info_p = power_plain(A, x0, eps=eps, max_iter=max_iter,
                                      stop_rule=stop_rule, log_each=log_each)
        display(Markdown(
            f"**итог (обычный):** итераций = **{info_p['iters']}**, "
            f"λ ≈ `{info_p['lambda']:.12g}`"
            + (f"  мяк: `{info_p.get('warn')}`" if info_p.get("warn") else "")
        ))
        display_latex(e_plain, label=r"\hat e_1 \;(\mathrm{plain})")
        display(Markdown(f"операции: `Q = {info_p['ops_total']['total']}`"))

        e_norm, info_n = power_normalized(A, x0, eps=eps, max_iter=max_iter,
                                          stop_rule=stop_rule, log_each=log_each)
        display(Markdown(
            f"**итог (нормировка):** итераций = **{info_n['iters']}**, "
            f"λ ≈ `{info_n['lambda']:.12g}`"
            + (f"  мяк: `{info_n.get('warn')}`" if info_n.get("warn") else "")
        ))
        display_latex(e_norm, label=r"\hat e_1 \;(\mathrm{normalized})")
        display(Markdown(f"операции: `Q = {info_n['ops_total']['total']}`"))

        display(Markdown("**мини-сравнение:**"))
        display(Markdown(
            f"обычный: итераций **{info_p['iters']}**, λ≈`{info_p['lambda']:.6g}`\n"
            f"нормир.: итераций **{info_n['iters']}**, λ≈`{info_n['lambda']:.6g}`"
        ))

        _compare_with_numpy(A, info_n["lambda"], e_norm)

    print("мяу, готово (=^..^=)")
def make_symmetric_with_spectrum(lambdas, seed=42):
    rng = np.random.default_rng(seed)
    lambdas = np.array(lambdas, dtype=float)
    n = lambdas.size
    M = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(M)
    A = Q @ np.diag(lambdas) @ Q.T
    return Mat(A)
def demo_three_inputs_power(n=6):
    display(Markdown("**1) быстрый пример (|λ1| сильно больше |λ2|)**"))
    A1 = make_symmetric_with_spectrum([5.0, 1.0] + [0.2]*(n-2), seed=7)
    x0 = make_x0_eig("rand", n, seed=1)
    solve_for_epsilons_power(A1, x0, eps_list=(1e-3, 1e-6), stop_rule="residual", log_each=True)

    display(Markdown("**2) медленный пример (|λ2/λ1| почти 1)**"))
    A2 = make_symmetric_with_spectrum([1.0, 0.99] + [0.2]*(n-2), seed=8)
    x0 = make_x0_eig("rand", n, seed=2)
    solve_for_epsilons_power(A2, x0, eps_list=(1e-3, 1e-6), stop_rule="residual", log_each=False)

    display(Markdown("**3) пример про переполнение/исчезновение порядка (λ1 очень большой)**"))
    A3 = make_symmetric_with_spectrum([50.0, 2.0] + [0.5]*(n-2), seed=9)
    x0 = make_x0_eig("rand", n, seed=3)
    solve_for_epsilons_power(A3, x0, eps_list=(1e-3, 1e-6), stop_rule="residual", log_each=False)
def _read_matrix_fallback():
    display(Markdown(
        "**ввод матрицы A:** вводи строки через пробел, пустая строка = конец.\n"
        "пример для 3x3:\n"
        "`1 2 3`\n`0 4 5`\n`0 0 6`"
    ))
    rows = []
    while True:
        s = input().strip()
        if s == "":
            break
        rows.append([float(t) for t in s.split()])
    if not rows:
        raise _err("пустой ввод матрицы")
    m = len(rows[0])
    if any(len(r) != m for r in rows):
        raise _err("строки разной длины")
    A = np.array(rows, dtype=float)
    return Mat(A)
def main():
    display(Markdown(
        "мяу мяу, что выбираем?\n\n"
        "[1] демка на трёх матрицах (быстро / медленно / очень большая λ1)\n\n"
        "[2] решить свою матрицу A (ручной ввод)\n\n"
        "[3] сгенерировать симметричную матрицу с заданным спектром\n"
    ))
    mode = (input("выбор (1/2/3): ").strip() or "1")

    if mode == "1":
        n = int(input("n (по умолчанию 6): ").strip() or "6")
        demo_three_inputs_power(n=n)
        return

    if mode == "2":
        try:
            from meow import _read_matrix
            A = _read_matrix()
        except Exception:
            A = _read_matrix_fallback()

        Aarr = _as_array_M(A)
        if Aarr.shape[0] != Aarr.shape[1]:
            raise _err("нужно ввести квадратную A")

        n = Aarr.shape[0]
        display(Markdown("выбираем x0: `ones` / `rand` / `e1`..`en`"))
        x0_kind = input(f"x0 kind (`ones` / `rand` / `e1`..`e{n}`) → ").strip() or "rand"
        x0 = make_x0_eig(x0_kind, n, seed=42)

        display(Markdown(
            "**выбираем критерий для остановки:**  \n"
            "`residual`: $\\sigma=\\dfrac{\\|Ax-\\lambda x\\|}{\\|x\\|}<\\varepsilon$  \n"
            "`lambda`: $\\big|\\lambda^{(k)}-\\lambda^{(k-1)}\\big|<\\varepsilon$"
        ))
        stop_rule = input("stop: `residual` / `lambda` [по умолчанию residual]: ").strip() or "residual"

        eps1 = float(input("eps для 1-го прогона (по умолчанию 1e-3): ").strip() or "1e-3")
        eps2 = float(input("eps для 2-го прогона (по умолчанию 1e-6): ").strip() or "1e-6")
        log_each = (input("печатать каждую итерацию? (y/n, по умолчанию y): ").strip().lower() or "y").startswith("y")

        solve_for_epsilons_power(A, x0, eps_list=(eps1, eps2),
                                 stop_rule=stop_rule, max_iter=500,
                                 log_each=log_each, show_true_eigs=True)
        return

    if mode == "3":
        n = int(input("n (по умолчанию 6): ").strip() or "6")
        seed = int(input("seed (по умолчанию 42): ").strip() or "42")
        display(Markdown(
            "задаём спектр (список λ через пробел)  \n"
            "если вводится меньше n чисел, оставшиеся заполняются 0.2"
        ))
        s = input(f"λ1..λm (по умолчанию: 5 1 {' '.join(['0.2']*(n-2))}): ").strip()
        if s == "":
            lambdas = [5.0, 1.0] + [0.2]*(n-2)
        else:
            lambdas = [float(t) for t in s.split()]
            if len(lambdas) < n:
                lambdas = lambdas + [0.2]*(n-len(lambdas))
            if len(lambdas) > n:
                lambdas = lambdas[:n]

        A = make_symmetric_with_spectrum(lambdas, seed=seed)

        display(Markdown("выбираем x0: `ones` / `rand` / `e1`..`en` (по умолчанию rand)"))
        x0_kind = input(f"x0 kind (`ones` / `rand` / `e1`..`e{n}`) → ").strip() or "rand"
        x0 = make_x0_eig(x0_kind, n, seed=seed+1)

        display(Markdown(
            "**выбираем критерий для остановки:**  \n"
            "`residual`: $\\sigma=\\dfrac{\\|Ax-\\lambda x\\|}{\\|x\\|}<\\varepsilon$  \n"
            "`lambda`: $\\big|\\lambda^{(k)}-\\lambda^{(k-1)}\\big|<\\varepsilon$"
        ))
        stop_rule = input("stop: `residual` / `lambda` [по умолчанию residual]: ").strip() or "residual"
        log_each = (input("печатать каждую итерацию? (y/n, по умолчанию n): ").strip().lower() or "n").startswith("y")

        solve_for_epsilons_power(A, x0, eps_list=(1e-3, 1e-6),
                                 stop_rule=stop_rule, max_iter=500,
                                 log_each=log_each, show_true_eigs=True)
        return

    print("мяу мяу, выбери 1/2/3")


if __name__ == "__main__":
    main()