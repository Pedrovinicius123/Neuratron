import sympy as sp

def explicar_resolucao(equacao):
    x = sp.Symbol('x')
    expr = sp.sympify(equacao)
    passos = sp.simplify(expr)

    return f"A equação {equacao} foi resolvida simplificando os termos: {passos}"