import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicTIB(nn.Module):
    def __init__(self, text_feat_dim, init_value=0.1, epsilon=1e-4):
        """
        Args:
            text_feat_dim (int): θ(t) 
            epsilon (float)
        """
        super(DynamicTIB, self).__init__()
        #self.init_value = min_temp
        self.epsilon = epsilon
        self.temp_predictor = nn.Linear(text_feat_dim, 1)
        self.temp_predictor.weight.data.fill_(0.0)
        self.temp_predictor.bias.data.fill_(math.log(init_value))
        print(self.temp_predictor.bias.data)
        

    def forward(self, text_features):
        """
        Args:
            text_features (torch.Tensor): [B, D_t]
        """
        # temp_scalar: [B, 1] -> [B]
        #print("weight:", self.temp_predictor.weight.data.shape, self.temp_predictor.weight.data)
        #print("bias:", self.temp_predictor.bias.data.shape, self.temp_predictor.bias.data)
        predicted_temp = self.temp_predictor(text_features).squeeze(-1)
        #print("predicted_temp:",predicted_temp)
        dynamic_pi = F.softplus(predicted_temp) + self.epsilon
        return dynamic_pi

# 注意：上面的代码假设 text_features 的维度 D_t 和 visual_features 的维度 D_v 可能不同，
# 因此通常需要一个投影层（例如 self.text_proj = nn.Linear(text_feat_dim, visual_feat_dim)）
# 请在 __init__ 中定义这个投影层。