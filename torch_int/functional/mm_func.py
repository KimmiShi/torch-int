import torch

from .._CUDA import (linear_a8_w8_b32_o32,
                     linear_relu_a8_w8_b8_o8,
                     linear_a8_w8_b8_o8,
                     linear_a8_w8_b32_o32_with_scaling,
                     linear_a8_w8_bfp32_ofp32,
                     bmm_s8t_s8n_f32t
                     )
from ..functional.quantization import (
    quantize_per_tensor_absmax,
    quantize_weight_per_channel_absmax,
    fake_quantize_activation_per_tensor_absmax,
    fake_quantize_activation_per_token_absmax,
)

class MatMula8w8b8o8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B, bias, s_b, s_c, s_o):
        # import pdb;pdb.set_trace()
        A, s_a = quantize_per_tensor_absmax(x)
        ctx.save_for_backward(A, B, bias)
        ctx.s_a=s_a
        ctx.s_b=s_b
        ctx.s_c=s_c
        ctx.x_grad = x.requires_grad

        x_shape = A.shape
        A = A.view(-1, x_shape[-1])
        y = linear_a8_w8_b8_o8(A, B, bias,
                               s_a*s_b/s_o, s_c/s_o)
        y = y.view(*x_shape[:-1], -1)


        return y

    # @staticmethod
    # def backward(ctx, grad_output):
    #     A, B, bias = ctx.saved_tensors
    #     empty_c = torch.tensor([0], dtype=torch.int8, device='cuda')
    #     q_grad_output, grad_scale = quantize_per_tensor_absmax(grad_output)
    #     grad_a=None
    #     grad_b=None
    #     grad_c=None
    #     if B.requires_grad:
    #         grad_b = linear_a8_w8_b8_o8(A.transpose(-1,-2), q_grad_output, empty_c, ctx.s_a*grad_scale, 0)
    #     if ctx.x_grad:
    #         grad_a = linear_a8_w8_b8_o8(q_grad_output, B.transpose(-1,-2), empty_c, ctx.s_b*grad_scale, 0)
    #     if bias.requires_grad:
    #         grad_c = torch.ones_like(bias)
    #     return grad_a, grad_b, grad_c, None, None, None

class MatMula8w8bfp32ofp32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, bias, s_w):
        # import pdb;pdb.set_trace()
        bias = bias.to(torch.float32)
        A, s_a = quantize_per_tensor_absmax(x)
        ctx.save_for_backward(A, W, bias)
        ctx.s_a=s_a
        ctx.s_w=s_w

        ctx.x_grad = x.requires_grad

        x_shape = A.shape
        A = A.view(-1, x_shape[-1])
        y = linear_a8_w8_bfp32_ofp32(A, W, bias,
                               s_a*s_w, 1)
        y = y.view(*x_shape[:-1], -1)

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # import pdb;pdb.set_trace()

        A, W, bias = ctx.saved_tensors
        # empty_c = torch.empty_like(bias)
        q_grad_output, grad_scale = quantize_per_tensor_absmax(grad_output)
        grad_a=None
        grad_b=None
        grad_c=None
        # if W.requires_grad:
        #     # grad_b = linear_a8_w8_bfp32_ofp32(A.transpose(-1,-2), q_grad_output, empty_c, ctx.s_a*grad_scale, 0)
        #     grad_b = bmm_s8t_s8n_f32t(A, q_grad_output, ctx.s_a*grad_scale)

        if ctx.x_grad:
            w_t = W.transpose(-1,-2).contiguous()
            # w2 = W.clone().detach()

            # import pdb;pdb.set_trace()
            # grad_a2 = bmm_s8t_s8n_f32t(q_grad_output, w2.unsqueeze(0), ctx.s_w*grad_scale)

            empty_c = torch.empty([A.shape[-1]], dtype=torch.float32, device='cuda')
            q_grad_output_shape = q_grad_output.shape
            q_grad_output = q_grad_output.view(-1, q_grad_output_shape[-1])
            grad_a = linear_a8_w8_bfp32_ofp32(q_grad_output, w_t, empty_c, ctx.s_w*grad_scale, 0)
            grad_a = grad_a.view(*A.shape[:-1], -1)
            # import pdb;pdb.set_trace()
        if bias.requires_grad:
            grad_c = torch.ones_like(bias)
        return grad_a, grad_b, grad_c, None
