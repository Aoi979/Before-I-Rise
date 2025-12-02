# fa_test.py
import torch
import torch.nn.functional as F
import time

def main():
    torch.manual_seed(42)

    # ====== 参数 ======
    B = 2        # batch size
    S = 128      # sequence length
    D = 64       # head dimension
    device = 'cuda'

    # ====== 生成输入 ======
    Q = torch.randn(B, S, D, device=device, dtype=torch.float32)
    K = torch.randn(B, S, D, device=device, dtype=torch.float32)
    V = torch.randn(B, S, D, device=device, dtype=torch.float32)

    # ====== 调用 PyTorch 的 Scaled Dot-Product Attention ======
    # PyTorch >= 2.1
    # Q,K,V shape: [batch, seq_len, head_dim]
    # 输出 shape: [batch, seq_len, head_dim]

    # 热身
    for _ in range(5):
        output_torch = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # 性能测试
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        output_torch = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"PyTorch Attention 100 iters: {elapsed:.4f} s")

    # ====== 输出示例和检查 ======
    print("Output shape:", output_torch.shape)
    print("Output sample (first batch, first seq token):", output_torch[0, 0, :5])

if __name__ == "__main__":
    main()
