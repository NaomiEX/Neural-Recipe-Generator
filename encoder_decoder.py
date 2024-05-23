import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from data import pack, unpack

from constants import MAX_INGR_LEN, DEVICE

def create_pretrained_embedding_dict(emb_filepath, **kwargs):
    ## build embedding dictionary
    pretrained_embedding_dict = dict()
    with open(emb_filepath, "r", **kwargs) as f:
        for line in f:
            wi, embi = line.rstrip().split(" ", 1)
            pretrained_embedding_dict[wi] = torch.from_numpy(np.fromstring(embi, sep=" "))
    return pretrained_embedding_dict

def create_pretrained_embeddings(pretrained_embedding_dict, input_size, embedding_size, vocab, verbose=True):
    """
    NOTE: input_size is typically len(vocab) although in decoder you don't need to generate
    unknown and pad.
    """
    emb = torch.randn([input_size, embedding_size])

    vocab_idxs_in_pretrained = []
    vocab_words = list(vocab._word2index.keys())
    for i in tqdm(range(input_size)):
        word = vocab_words[i]
        if word in pretrained_embedding_dict:
            vocab_idxs_in_pretrained.append(i)
            emb[i] = pretrained_embedding_dict[word]
    
    if verbose: 
        print(f"{len(vocab_idxs_in_pretrained)}/{input_size} ({len(vocab_idxs_in_pretrained)/input_size:.3f}%) words have pretrained embeddings")

    return emb, vocab_idxs_in_pretrained

def create_embeddings(input_size, embedding_size, pretrained_embedding_dict, vocab):
    if pretrained_embedding_dict is None:
        return nn.Embedding(input_size, embedding_size), None
    else:
        assert vocab is not None
        embeddings_w_pretrained, vocab_idxs_in_pretrained = create_pretrained_embeddings(
            pretrained_embedding_dict, input_size, embedding_size, vocab
        )
        # ! For now not freezing embeddings because ~40% don't exist in pretrained embeddings
        # plus allows for finetuning
        return nn.Embedding.from_pretrained(embeddings_w_pretrained, freeze=False), vocab_idxs_in_pretrained

class EncoderRNN(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 hidden_size,
                 padding_value,
                 pretrained_embedding_dict=None,
                 vocab=None,
                 ):
        """Encoder LSTM to encode input sequence.

        input_size (int): size of vocabulary
        hidden_size (int): size of hidden dimension, referred to as H
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding, self.pretrained_emb_idxs = create_embeddings(input_size, embedding_size, pretrained_embedding_dict, vocab)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.padding_value = padding_value

    def update_embedding_grad(self, grads):
        if not hasattr(self, "pretrained_emb_idxs"):
            return
        
        grads[self.pretrained_emb_idxs] = 0 # zero-out gradients for pretrained embeddings

    def forward(self, ingredients, ing_lens):
        """Embed ingredients and feed through LSTM. 
        Batch process all words in sequence at once for efficiency rather than one word one batch at a time.

        Args:
            ingredients (torch.Tensor): padded ingredients of shape [N, L], where N=batch size and L=longest sequence length in batch
        """
        ## embed ingredients
        ingredients_embed = self.embedding(ingredients) # [N, L, E]

        ## pack padded ingredients tensor before feeding through LSTM (this allows the lstm to optimize operations, ignoring padding)
        ingredients_packed = pack(ingredients_embed, ing_lens)

        ## feed through LSTM
        # by default, initial hidden state and initial cell state are zeros
        # output: PackedSequence containing hidden state for each token in sequence
        # final hidden state: Tensor [num_layers=1, N, H] NOTE: this is the last non-padded hidden state for each input sequence
        # c_final: last cell state Tensor [num_layers=1, N, H]
        output, (h_final, c_final) = self.lstm(ingredients_packed)

        ## unpack PackedSequence to get back our padded tensor
        # output_padded: padded output tensor which masks out encoder outputs for padding to 0; shape [N, L, H] NOTE: output_padded[:, -1] != h_final because of padding
        # output_lens: unpadded sequence lengths; tensor of shape [N]
        output_padded, output_lens = unpack(output, padding_val=self.padding_value)

        return output_padded, output_lens, h_final, c_final
    
#! IMPORTANT: MAKE SURE DECODER'S OUTPUT SIZE IS VOCAB SIZE - 1
class DecoderRNN(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 output_size,
                 pretrained_embedding_dict=None,
                 vocab=None,
                 ):
        """Decoder to generate recipes based on encoder output (hidden state(s)).

        Args:
            hidden_size (int): size of hidden dimension
            output_size (int): size of target language vocabulary - 1 (doesn't need to encode padding), |Vocab| - 1
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding, self.pretrained_emb_idxs = create_embeddings(output_size, embedding_size, pretrained_embedding_dict, vocab)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=False)
        self.nonlinear_activation = nn.Tanh()
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def update_embedding_grad(self, grads):
        if not hasattr(self, "pretrained_emb_idxs"):
            return
        
        grads[self.pretrained_emb_idxs] = 0 # zero-out gradients for pretrained embeddings

    def forward(self, inp, hidden, cell):
        """Decode one word at a time. Batch processed.

        Args:
            inp (torch.Tensor): start token or previous generation (non teacher-forcing) or 
                                previous ground truth token (teacher-forcing);
                                shape [N]
            hidden (torch.Tensor): encoder/decoder last hidden state; shape [1, N, H]
            cell (torch.Tensor): encoder/decoder last hidden state; shape [1, N, H]
        """
        ## embed token input
        # if val and ((inp > self.embedding.num_embeddings).sum() > 0):
        #     inp_embedded = torch.zeros([inp.shape[0], self.embedding.embedding_dim], device=DEVICE)
        #     mask = inp < self.embedding.num_embeddings
        #     inp_embedded[mask] = self.embedding(inp[mask])
        #     inp_embedded = inp_embedded[None]
        # else:
        inp_embedded = self.embedding(inp)[None] # [L=1, N, E]

        ## apply non-linear activation
        inp_embedded = self.nonlinear_activation(inp_embedded)
        
        ## feed embedded input and hidden state through LSTM
        # out: output features; shape [L=1, N, H]
        # h_final: final updated hidden state; shape [num_layers=1, N, H]
        # c_final: last cell state Tensor [num_layers=1, N, H]
        out, (h_final, c_final) = self.lstm(inp_embedded, (hidden, cell))

        ## linear projection
        out = self.out_fc(out[0]) # [N, H] -> [N, |Vocab|]

        ## log softmax to get log probability distribution over vocabulary words
        out = self.logsoftmax(out) # [N, |Vocab|]

        return out, h_final, c_final
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, padding_val,
                 dropout=0.1, global_max_ing_len=MAX_INGR_LEN,
                 pretrained_embedding_dict=None,
                 vocab=None,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.padding_val = padding_val
        self.global_max_ing_len = global_max_ing_len

        self.embedding, self.pretrained_emb_idxs = create_embeddings(output_size, embedding_size, 
                                                                     pretrained_embedding_dict, vocab)
        self.attn = nn.Linear(hidden_size + embedding_size, global_max_ing_len)
        self.attn_combine = nn.Linear(hidden_size + embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=False)
        self.nonlinear_activation = nn.ReLU() # TODO: TRY TANH
        self.out_fc = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def update_embedding_grad(self, grads):
        if not hasattr(self, "pretrained_emb_idxs"):
            return
        
        grads[self.pretrained_emb_idxs] = 0 # zero-out gradients for pretrained embeddings

    def mask_attn_weights(self, ingredients, attn_weights):
        """
        ingredients: [N, L_i]
        """
        # ing_len = ingredients.size(1)
        # pad ingredients on the right with `padding_val` to maximum len
        # [N, max_len]
        # padd_maxlen_ingredients = F.pad(
        #     ingredients, (0, self.global_max_ing_len-ing_len), value=0)
        # [N, L_i] where 1 is masked value, 0 is valid value
        attn_mask = ingredients == self.padding_val
        assert list(attn_weights.shape) == list(attn_mask.shape)
        attn_weights[attn_mask] = -torch.inf # set as -inf because when softmax-ed will turn to 0
        return attn_weights

    def forward(self, inp, hidden, cell, encoder_houts, ingredients):
        """
        Args:
            inp (torch.Tensor): start token or previous generation (non teacher-forcing) or 
                                previous ground truth token (teacher-forcing);
                                shape [N]
            hidden (torch.Tensor): encoder last hidden state; shape [1, N, H]
            cell (torch.Tensor): encoder last cell state; shape [1, N, H]
            encoder_houts (torch.Tensor): encoder all hidden states for all elements in sequence;
                                          padded tensor [N, L_i, H], where L_i is the max sequence len
        """
        L_i = encoder_houts.size(1) # max seq len in this batch
        ## embed token input
        inp_embedded = self.embedding(inp) # [N, E]

        inp_embedded = self.dropout(inp_embedded)

        attn_weights = self.attn(
            torch.cat((inp_embedded, hidden[0]), dim=1) # [N, E+H]
        )[:, :L_i] # [N, max_ing_len] -> [N, L_i]
        # [N, L_i]
        attn_weights = F.softmax(
            self.mask_attn_weights(ingredients, attn_weights),
            dim=-1)
        # [N, 1, L_i] bmm [N, L_i, H] = [N, 1, H] -> [N, H]
        attn_res = torch.bmm(attn_weights[:, None], encoder_houts)[:, 0]

        # [N, E+H]
        output = torch.cat((inp_embedded, attn_res), dim=-1)
        # [N, E+H] -> [N, H] -> [L=1, N, H]
        output = self.attn_combine(output)[None]
        output = self.nonlinear_activation(output)

        ## feed embedded input and hidden state through LSTM
        # out: output features; shape [L=1, N, H]
        # h_final: final updated hidden state; shape [num_layers=1, N, H]
        # c_final: last cell state Tensor [num_layers=1, N, H]
        out, (h_final, c_final) = self.lstm(output, (hidden, cell))

        out = self.out_fc(out) # [L=1, N, H] -> [L=1, N, |Vocab|]

        ## log softmax to get log probability distribution over vocabulary words
        out = self.logsoftmax(out[0])

        return out, h_final, c_final, attn_weights
    
class CoverageAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, padding_val,
                 dropout=0.1, global_max_ing_len=MAX_INGR_LEN,
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.padding_val = padding_val
        self.global_max_ing_len = global_max_ing_len

        self.embedding, self.pretrained_emb_idxs = create_embeddings(output_size, embedding_size,
                                                                     None, None)
        self.dropout = nn.Dropout(dropout)
        
    def update_embedding_grad(self, grads):
        if not hasattr(self, "pretrained_emb_idxs"):
            return
        
        grads[self.pretrained_emb_idxs] = 0 # zero-out gradients for pretrained embeddings

    def forward(self, inp, hidden, cell, encoder_houts, ingredients):
        """_summary_

        Args:
            inp (torch.Tensor): start token or previous generation (non teacher-forcing) or 
                                previous ground truth token (teacher-forcing);
                                shape [N]
            hidden (torch.Tensor): encoder last hidden state; shape [1, N, H]
            cell (torch.Tensor): encoder last cell state; shape [1, N, H]
            encoder_houts (torch.Tensor): encoder all hidden states for all elements in sequence;
                                          padded tensor [N, L_i, H], where L_i is the max sequence len
            ingredients (torch.Tensor): padded ingredients, used to determine padding mask;
                                        shape [N, L_i], where L_i = max ingredients length in batch
        """
        L_i = encoder_houts.size(1) # max seq len in this batch
        ## embed token input
        inp_embedded = self.embedding(inp) # [N, E]

        inp_embedded = self.dropout(inp_embedded)