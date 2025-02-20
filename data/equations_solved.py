import sympy as sp

def resolver_equacao(equacao):
    x = sp.Symbol('x')
    expr = sp.sympify(equacao)
    solucao = sp.solve(expr, x)
    return solucao
    
