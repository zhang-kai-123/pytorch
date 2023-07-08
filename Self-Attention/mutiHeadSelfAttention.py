import torch  # 导入PyTorch库
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  # 继承自nn.Module基类
        self.n_heads = n_heads  # 多头注意力头数
        self.d_model = d_model  # 输入向量维度
        self.d_k = d_model // n_heads  # 每个头的维度
        self.dropout = nn.Dropout(p=dropout)  # dropout概率

        # 初始化Query、Key、Value的权重矩阵
        self.W_q = nn.Linear(d_model, n_heads * self.d_k)  # Query权重矩阵
        self.W_k = nn.Linear(d_model, n_heads * self.d_k)  # Key权重矩阵
        self.W_v = nn.Linear(d_model, n_heads * self.d_k)  # Value权重矩阵

        # 初始化输出的权重矩阵
        self.W_o = nn.Linear(n_heads * self.d_k, d_model)  # 输出向量的权重矩阵

    def forward(self, x, mask=None):
        # 输入 x 的维度为 [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape  # size()和 shape输出结果相同，与size输出不同；

        # 通过权重矩阵计算 Q、K、V 将输入张量沿着最后一维进行分割，生成多个头（multi-heads）所需要的查询、键、值矩阵
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # 交换维度以便于计算注意力权重
        Q = Q.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)
        K = K.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)
        V = V.permute(0, 2, 1, 3).contiguous().view(batch_size * self.n_heads, seq_len, self.d_k)

        # 计算注意力权重
        scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_weights = self.dropout(attn_weights)

        # 计算输出向量
        attn_output = torch.bmm(attn_weights, V)
        attn_output = attn_output.view(batch_size, self.n_heads, seq_len, self.d_k)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len,
                                                                        self.n_heads * self.d_k)
        output = self.W_o(attn_output)

        return output


# 定义输入向量
x = torch.randn(2, 10, 128)

# 定义注意力模块
attn = MultiHeadAttention(n_heads=8, d_model=128)

# 进行前向传播计算
output = attn(x)

# 打印输出向量的形状
print(output.shape)  # 输出：torch.Size([2, 10, 128])
