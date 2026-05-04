import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MotionEncoderBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoderBiGRU, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(seq_len, batch_size, dim)
    def forward(self, inputs, m_lens, *args, **kwargs):
        # Transpose the input tensor to (B, T, D)
        inputs = inputs.permute(1, 0, 2)
        # Sort the batch by sequence length in descending order (requirement for pack_padded_sequence)
        sorted_lengths, sorted_indices = torch.sort(m_lens, descending=True)
        inputs_sorted = inputs[sorted_indices]
        
        num_samples = inputs_sorted.shape[0]
        input_embs = self.input_emb(inputs_sorted)
        hidden = self.hidden.repeat(1, num_samples, 1)

        # Convert sorted lengths to a list for the packing function
        cap_lens = sorted_lengths.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        _, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        output = self.output_net(gru_last)

        # Restore the original order of the batch before returning
        _, original_indices = torch.sort(sorted_indices, descending=False)
        output_restored = output[original_indices]

        return output_restored


class MotionEncoderAttentionBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MotionEncoderAttentionBiGRU, self).__init__()

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Attention network is now a permanent part of the model
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, 1)
        )

        # Output network processes the result of the attention mechanism
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        # Apply initialization
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.attention_net.apply(init_weight)
            
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        """
        inputs: (seq_len, batch_size, dim) - Note: code permutes to batch_first
        """
        inputs = inputs.permute(1, 0, 2)
        sorted_lengths, sorted_indices = torch.sort(m_lens, descending=True)
        inputs_sorted = inputs[sorted_indices]
        
        num_samples = inputs_sorted.shape[0]
        input_embs = self.input_emb(inputs_sorted)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = sorted_lengths.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, _ = self.gru(emb, hidden) # BiGRU forward
        gru_seq, _ = pad_packed_sequence(gru_seq, batch_first=True)
        attention_scores = self.attention_net(gru_seq).squeeze(-1) # Attention scores

        batch_size, max_len = gru_seq.shape[:2]
        mask = torch.arange(max_len, device=gru_seq.device).expand(
            batch_size, max_len
        ) < sorted_lengths.unsqueeze(1)
        
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1) # Shape: (batch, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), gru_seq).squeeze(1)
        output = self.output_net(context_vector) # final hidden state after attention

        # restore original order
        _, original_indices = torch.sort(sorted_indices)
        output = output[original_indices]
        attention_weights = attention_weights[original_indices]
        
        return output, attention_weights
    