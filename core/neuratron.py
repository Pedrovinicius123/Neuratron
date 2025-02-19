import numpy as np
import torch
import torch.nn.functional as F
from torch.sparse import coo_tensor
from sklearn.metrics import accuracy_score
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

class ArtificialBrain:
    def __init__(self, size, version="1.8"):
        self.size = size
        self.version = version
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        indices = torch.randint(0, size, (2, size * 10), device=self.device)  # Matriz esparsa
        values = (torch.randn(size * 10, dtype=torch.float16, device=self.device) * 0.1)
        self.weights = coo_tensor(indices, values, (size, size), requires_grad=True).to(self.device)
        
        self.activation_matrix = torch.ones(size, size, dtype=torch.float16, device=self.device)
        torch.diagonal(self.activation_matrix).zero_()
        self.performance_history = []  # Histórico de desempenho para metacognição
        self.emotional_state = torch.zeros(size, dtype=torch.float16, device=self.device)  # Estados emocionais
        
        # Inicialização do modelo de PLN
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
    
    @lru_cache(maxsize=128)
    def generate_activation_matrix(self, input_vector):
        """Gera uma matriz de ativação baseada no input."""
        activation_values = torch.sigmoid(input_vector.to(self.device))
        activation_matrix = torch.outer(activation_values, activation_values).to(dtype=torch.float16, device=self.device)
        torch.diagonal(activation_matrix).zero_()
        return activation_matrix
    
    def forward(self, input_vector):
        """Propaga o input através do cérebro artificial."""
        input_vector = input_vector.to(self.device)
        self.activation_matrix = self.generate_activation_matrix(tuple(input_vector.tolist()))
        activated_input = self.activation_matrix @ input_vector
        output = F.relu(self.weights @ activated_input)
        self.update_emotional_state(output)  # Atualiza o estado emocional
        return output.cpu()      
    
    def update_weights(self, input_vector, target, lr=0.01):
        """Atualiza os pesos considerando a matriz de ativação e erro."""
        input_vector, target = input_vector.to(self.device), target.to(self.device)
        output = self.forward(input_vector)
        loss = F.mse_loss(output, target)
        loss.backward()
        with torch.no_grad():
            self.weights.values().sub_(lr * self.weights.grad.values())
            self.weights.grad = None
        self.track_performance(loss.item())  # Registro de desempenho
    
    def update_emotional_state(self, output):
        """Ajusta os estados emocionais com base na saída."""
        self.emotional_state = torch.tanh(output.mean())  # Emulação simples de emoção
