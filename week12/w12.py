import torch

torch.manual_seed(2023)

def activation_func(x):
    # ReLU (bạn có thể đổi thành sigmoid, tanh, hoặc leaky ReLU)
    return torch.relu(x)

def softmax(x):
    # Softmax: tính phân phối xác suất
    exp_x = torch.exp(x - torch.max(x))  # Trừ max để tránh tràn số
    return exp_x / exp_x.sum(dim=1, keepdim=True)

# Define the size of each layer in the network
num_input = 784
num_hidden_1 = 128
num_hidden_2 = 256
num_hidden_3 = 128
num_classes = 10

# Random input
input_data = torch.randn((1, num_input))
W1 = torch.randn(num_input, num_hidden_1)
W2 = torch.randn(num_hidden_1, num_hidden_2)
W3 = torch.randn(num_hidden_2, num_hidden_3)
W4 = torch.randn(num_hidden_3, num_classes)
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

z1 = input_data @ W1 + B1
a1 = activation_func(z1)
z2 = a1 @ W2 + B2
a2 = activation_func(z2)
z3 = a2 @ W3 + B3
a3 = activation_func(z3)
z4 = a3 @ W4 + B4
result = softmax(z4)

print("Output:", result)
print("Sum of output:", result.sum().item())
