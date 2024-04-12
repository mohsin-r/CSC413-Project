# For any functions both models may require or share

import torch

# Take a tensor "data", generate "seq_len"-long sequences each starting "gap" indices apart
# Then assign the next value the target of the sequence
def generate_sequences(data: torch.Tensor, seq_len: int, gap: int) -> tuple[torch.Tensor, torch.Tensor]:

    num_sequences = int((data.size(dim=0) - seq_len) / gap) + 1
    out_seqs = torch.empty(num_sequences, seq_len, 1)
    targets = torch.empty(num_sequences, 1)

    for i in range(num_sequences):
        out_seqs[i] = torch.unsqueeze(data[gap * i : gap * i + seq_len], dim=1)
        targets[i] = torch.unsqueeze(data[gap * i + seq_len], dim=0)
    
    return out_seqs, targets