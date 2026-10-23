import torch
import time
from model import MyMiniGPT
with open('my_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"你的文本总长度:{len(text)} 个字符")
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i,ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
data = torch.tensor(encode(text), dtype= torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[:n]
block_sie = min(32, len(data) -1)
batch_size = 8
n_embd = 64
n_head=4
n_layer=2
max_iters=200
device='cpu'
def get_batch(split):
        data_batch=train_data if split == 'train' else val_data
        max_start_idx=max(1,len(data_batch) - batch_size)
        ix = torch.randint(max_start_idx, (batch_size,))
        x = torch.stack([data_batch[i:i+batch_size] for i in ix])
        y = torch.stack([data_batch[i+1:i+batch_size+1] for i in ix])
        return x.to(device), y.to(device)
print(f"正在使用设备；{device}")
model=MyMiniGPT(vocab_size, n_embd, block_sie, n_head, n_layer).to(device)
optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
print("\n--- 开始训练 ---")
for iter in range(max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iter % 50 == 0 or iter == max_iters -1:
            print(f"Step {iter}:Loss ={loss.item():.4f}")
model.eval()
with torch.no_grad():
    xb, yb = get_batch('val')
    _, val_loss = model(xb, yb)
    print(f"\nright 验收指标 1-验证集推理误差 (Val Loss): {val_loss.item():.4f}")
print("\n--- 开始推理吞吐能力压测 ---")
start_time = time.time()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = 100
with torch.no_grad():
    for _ in range(generated_tokens):
        context_crop = context[:, -block_sie:]
        logits, _ = model(context_crop)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)
end_time = time.time()
tokens_per_sec = generated_tokens / (end_time - start_time)
print(f"right 验收指标 2 -推理吞吐能力(Tokens/sec): {tokens_per_sec:.2f} tokens/秒")


