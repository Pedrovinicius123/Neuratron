import torch
import torch.nn as nn
import torch.optim as optim

class MatrizCerebral(nn.Module):
    def __init__(self, num_neuronios):
        super(MatrizCerebral, self).__init__()
        
        # Matriz de conexões neurais (W) - treinável
        self.W = nn.Parameter(torch.randn(num_neuronios, num_neuronios))
        
        # Matriz de ativação (A) - valores entre 0 e 1
        self.A = nn.Parameter(torch.rand(num_neuronios, num_neuronios))
          

    def forward(self, x):
        # Multiplicação da matriz de ativação A pela matriz de pesos W
        conexoes_ativas = self.A * self.W  
        
        # Passagem dos dados pela rede
        x = torch.matmul(conexoes_ativas, x)  
        
        return torch.relu(x)  # Ativação não linear