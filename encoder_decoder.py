import torch
from torch import nn
from data import pack, unpack

class EncoderRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 padding_value,
                 ):
        """Encoder LSTM to encode input sequence.

        input_size (int): size of vocabulary
        hidden_size (int): size of hidden dimension, referred to as H
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.padding_value = padding_value

    def forward(self, ingredients, ing_lens):
        """Embed ingredients and feed through LSTM. 
        Batch process all words in sequence at once for efficiency rather than one word one batch at a time.

        Args:
            ingredients (torch.Tensor): padded ingredients of shape [N, L], where N=batch size and L=longest sequence length in batch
        """
        ## embed ingredients
        ingredients_embed = self.embedding(ingredients) # [N, L, H]

        ## pack padded ingredients tensor before feeding through LSTM (this allows the lstm to optimize operations, ignoring padding)
        ingredients_packed = pack(ingredients_embed, ing_lens)

        ## feed through LSTM
        # by default, initial hidden state and initial cell state are zeros
        # output: PackedSequence containing hidden state for each token in sequence
        # final hidden state: Tensor [num_layers=1, N, H] NOTE: this is the last non-padded hidden state for each input sequence
        # c_final: last cell state Tensor [num_layers=1, N, H]
        output, (h_final, _) = self.lstm(ingredients_packed)

        ## unpack PackedSequence to get back our padded tensor
        # output_padded: padded output tensor which masks out encoder outputs for padding to 0; shape [N, L, H] NOTE: output_padded[:, -1] != h_final because of padding
        # output_lens: unpadded sequence lengths; tensor of shape [N]
        output_padded, output_lens = unpack(output, padding_val=self.padding_value)

        return output_padded, output_lens, h_final
    
#! IMPORTANT: MAKE SURE DECODER'S OUTPUT SIZE IS VOCAB SIZE - 1
class DecoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size
                 ):
        """Decoder to generate recipes based on encoder output (hidden state(s)).

        Args:
            hidden_size (int): size of hidden dimension
            output_size (int): size of target language vocabulary - 1 (doesn't need to encode padding), |Vocab| - 1
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.nonlinear_activation = nn.Tanh()
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        """Decode one word at a time. Batch processed.

        Args:
            inp (torch.Tensor): start token or previous generation (non teacher-forcing) or 
                                previous ground truth token (teacher-forcing);
                                shape [N]
            hidden (torch.Tensor): encoder last hidden state; shape [1, N, H]
        """
        ## embed token input
        inp_embedded = self.embedding(inp)[None] # [L=1, N, H]

        ## apply non-linear activation
        inp_embedded = self.nonlinear_activation(inp_embedded)
        
        ## feed embedded input and hidden state through LSTM
        # out: output features; shape [L=1, N, H]
        # h_final: final updated hidden state; shape [num_layers=1, N, H]
        # c_final: last cell state Tensor [num_layers=1, N, H]
        out, (h_final, _) = self.lstm(inp_embedded, (hidden, torch.zeros_like(hidden)))

        ## linear projection
        out = self.out_fc(out[0]) # [N, H] -> [N, |Vocab|]

        ## log softmax to get log probability distribution over vocabulary words
        out = self.logsoftmax(out) # [N, |Vocab|]

        return out, h_final