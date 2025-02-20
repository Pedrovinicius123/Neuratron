import torch
import torch.nn as nn

def treinar_modelo(modelo, epocas:int):
    # Função de erro e otimizador
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(modelo.parameters(), lr=0.001)

    # Treinamento
    for epoch in range(epocas):
        for entrada, saida_esperada in dados[:1000]:  # Pegamos 1000 exemplos para cada época
            optimizer.zero_grad()
            saida_predita = modelo(entrada)
            loss = loss_fn(saida_predita, saida_esperada)
            loss.backward()
            optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Época {epoch}, Erro: {loss.item()}")