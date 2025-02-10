import torch

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""
# 1. Tensor 3D kích thước 20x30x40 toàn 0
tensor_3d = torch.zeros(20, 30, 40)
print(tensor_3d)
# 2. Tensor 1D chứa số chẵn từ 10 đến 100
tensor_even = torch.arange(10, 101, step=2)
print(tensor_even)

"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
x = torch.rand(4, 6)  # Tạo tensor ngẫu nhiên
print(x)
# 1. Tổng toàn bộ phần tử
total_sum = torch.sum(x)
print(total_sum)

col_sum = torch.sum(x, dim=0)
print(col_sum)

row_sum = torch.sum(x, dim=1)
print(row_sum)

"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
x = torch.tensor([0.1, 0.3, 2.3, 0.45])
y = torch.tensor([0.13, 0.23, 2.33, 0.45])
dot_product = torch.dot(x, y)
print(f"dot_product: {dot_product}")
norm_x = torch.norm(x)
print(f"norm_x: {norm_x}")
norm_y = torch.norm(y)
print(f"norm_y: {norm_y}")
cosine_similarity = torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
print(f"cosine_similarity: {cosine_similarity}")

"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                  [-2.2268, 1.9799, 1.5682, 0.5850],
                  [ 1.2289, 0.5043, -0.1625, 1.1403]])
y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                  [-0.6679, 0.0793, -2.5842, -1.5123],
                  [ 1.1110, -0.1212, 0.0324, 1.1277]])

x_reshaped = x.reshape(-1)
y_reshaped = y.reshape(-1)
cosine_similarity_2 = torch.dot(x_reshaped, y_reshaped) / (torch.norm(x_reshaped) * torch.norm(y_reshaped))
print(cosine_similarity_2)


"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
x = torch.tensor([[ 0,  1],
                  [ 2,  3],
                  [ 4,  5],
                  [ 6,  7],
                  [ 8,  9],
                  [10, 11]])
x_1d = x.view(-1)
print(x_1d)
x_2d = x_1d.view(3, 4)
print(x_2d)

"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
x = torch.rand(3, 1080, 1920)
print(x)
y = torch.rand(3, 720, 1280)

x_4d = x.unsqueeze(0)
y_4d = y.unsqueeze(0)

y_resized = torch.nn.functional.interpolate(y_4d, size=(1080, 1920), mode='bilinear')
print(y_resized.shape)

combined = torch.cat((x_4d, y_resized), dim=0)
print(combined.shape)


