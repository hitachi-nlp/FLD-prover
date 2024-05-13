""" 
    PyTorch Mixtral model MixtralSparseMoeBlock patch for Deepspeed Zero3.
    Tested for transformers==4.36.2.
    See [here](https://github.com/hiyouga/LLaMA-Factory/issues/2315)
"""

import math
import random

import unittest
import torch
import torch.nn.functional as F

def mlp_forward(self, hidden_states):
    current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
    current_hidden_states = self.w2(current_hidden_states)
    return current_hidden_states


## Ref. https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    # router_logits: (batch * sequence_length, n_experts)
    router_logits = self.gate(hidden_states)

    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    topk_weight, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
    topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
    # we cast back to the input dtype
    topk_weight = topk_weight.to(hidden_states.dtype)

    hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
    y = torch.empty_like(hidden_states)
    flat_topk_idx = topk_idx.view(-1)
    for i in range(self.num_experts):
        expert = self.experts[i]
        y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
    y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
    final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def replace_mixtral_moe_with_dense_impl():
    from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
    from transformers.models.mixtral.modeling_mixtral import MixtralBLockSparseTop2MLP

    MixtralBLockSparseTop2MLP.forward = mlp_forward
    MixtralSparseMoeBlock.forward = moe_forward


class TestMixtralSparseMoeBlock(unittest.TestCase):
    def setUp(self):
        from transformers.models.mixtral.configuration_mixtral import MixtralConfig
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        model_name_or_path = 'PATH_XXX'
        import transformers
        self.config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
        )
        self.config.use_cache = False
        self.config.output_router_logits = True
        self.device = "cuda"
        self.model = MixtralSparseMoeBlock(self.config)
        self.model.to(self.device)

    def test_forward_pass(self):
        batch_size = 1
        seqlen = 512
        hidden_size = self.config.hidden_size
        hidden_states = torch.randn(batch_size, seqlen, hidden_size, device=self.device)
        with torch.no_grad():
            output1, _ = self.model(hidden_states)
            replace_mixtral_moe_with_dense_impl()
            output2, _ = self.model(hidden_states)
        rtol = 1e-3
        atol = ((output1 + 0.3 - 0.3) - output1).abs().max().item()
        assert torch.allclose(output1, output2, rtol=rtol, atol=2 * atol)


if __name__ == '__main__':
    unittest.main()
