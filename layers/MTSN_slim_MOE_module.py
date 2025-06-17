import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

class TimeMoeTemporalBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    

class MoeSparseExpertsLayer(nn.Module):
    def __init__(self, config, hidden_size, intermediate_size):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.norm_topk_prob = False

        moe_intermediate_size = intermediate_size // self.top_k

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TimeMoeTemporalBlock(
                hidden_size=self.hidden_size,
                intermediate_size=moe_intermediate_size,
                hidden_act=self.config.hidden_act,
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = TimeMoeTemporalBlock(
            hidden_size=self.hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=self.config.hidden_act,
        )
        self.shared_expert_gate = torch.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        # router_logits -> (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    

def load_balancing_loss_func(
        gate_logits: torch.Tensor, # 传入的gate_logits可以是一个tensor，也可以是一个tuple或者list
        # 如果是tensor则要求形状为(batch_size*seq_len, num_experts)
        top_k: int,
        num_experts: int = None,
        ) -> torch.Tensor:

        if gate_logits is None: # 检查gate_logits是否有效，无效则返回0.0
            return 0.0

        # Concat the logits from all layers
        # compute_device = gate_logits[0].device

        # concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

        routing_weights = torch.nn.functional.softmax(gate_logits, dim=-1) # 最后一维应用softmax

        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1) # 选取top_k个专家

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts) # 将top_k个专家的索引转换为one-hot编码

        # 计算tok_k每次选择中选择每个专家的路由概率
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # 综合top_k次选择中每个专家的路由概率，由routing_weights的均值得到，给出了门控单元给予每个专家的概率
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
        probs = router_prob_per_expert.detach().cpu().numpy()

        overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0)) # 最终损失的计算通过广播机制实现

        return overall_loss * num_experts, probs