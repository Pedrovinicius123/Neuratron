import sympy as sp
import random
import torch

# Criar um conjunto de equações simples para treino
dados = []
for _ in range(10000):  # Gerar 10.000 exemplos
    a = random.randint(1, 10)
    b = random.randint(-10, 10)
    x = sp.Symbol('x')
    equacao = a * x + b
    solucao = sp.solve(equacao, x)
    
    # Criar entrada e saída para treino
    entrada = torch.tensor([a, b], dtype=torch.float32)
    saida = torch.tensor(solucao, dtype=torch.float32)

    dados.append((entrada, saida))
