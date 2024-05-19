import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import pad_collate

## constants
from data import PAD_WORD
TEACHER_FORCING_RATIO = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##

def train_iter(ingredients, recipes, ing_lens, rec_lens, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
               vocab=None # !remove later
               ):
    """Single training iteration. Processes batched data.

    Args:
        ingredients (torch.Tensor): padded ingredients tensor in idx form; 
                                    shape [N, L_i], where L_i = max ingredients length in batch
        recipes (torch.Tensor): padded recipes tensor in idx form;
                                shape [N, L_r], where L_r = max recipes length in batch
        ing_lens (torch.Tensor): unpadded length of ingredients; shape [N]
        rec_lens (torch.Tensor): unpadded length of recipes; shape [N]
        encoder (EncoderRNN): encoder RNN module
        decoder (DecoderRNN): decoder RNN module
        encoder_optimizer (torch.optim)
        decoder_optimizer (torch.optim)
        criterion (torch.nn.NLLLoss): loss function
    """

    ## reset gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    N = ingredients.size(0)
    padded_ing_len = ingredients.size(1) # L_i
    padded_rec_len = recipes.size(1) # L_r


    ## feed ingredients through encoder
    # enc_out: padded encoder output tensor with shape [N, L, H]
    # enc_out_lens: unpadded sequence lengths; tensor with shape [N]
    # enc_h_final: final hidden state: [num_layers=1, N, H]
    enc_out, enc_out_lens, enc_h_final = encoder(ingredients, ing_lens)

    # decoder_input = torch.full((N, 1), fill_value=vocab.word2index["<RECIPE_START>"],
    #                            dtype=torch.long, device=DEVICE) 
    # initialize decoder hidden state as final encoder hidden state
    decoder_hidden = enc_h_final

    if TEACHER_FORCING_RATIO < 1:
        raise ValueError("Non-teacher forcing is not implemented")
    
    loss = 0
    all_decoder_outs = [] # List of [N, |Vocab|-1]
    all_gt = [] # List of [N]

    ## teacher forcing
    curr_rec_lens = rec_lens.clone()
    ## NOTE: recipes already contain start token no need to add manually
    ## TODO IMPORTANT: MAKE SURE DECODER'S OUTPUT SIZE IS VOCAB SIZE - 1
    for di in range(padded_rec_len-1):
        # get batches which have valid (non-padding and non ending) tokens as input
        valid = (rec_lens - 1) > di
        decoder_input_i = recipes[valid, di] # [N_valid]
        decoder_hidden_i = decoder_hidden[:,valid] # [1, N_valid, H]

        # decoder_out: log probabilities over vocab; [N_valid, |Vocab|-1]
        # decoder_hfinal: final hidden state; [num_layers=1, N_valid, H]
        decoder_out, decoder_hidden_i = decoder(decoder_input_i, decoder_hidden_i)

        all_decoder_outs.append(decoder_out)

        # because we ensured that input cannot be end token, there is a guaranteed non-padding token
        # for each valid batch sample
        gt_i = recipes[valid, di+1] # [N_valid]
        assert (gt_i != vocab.word2index[PAD_WORD]).all(), f"gt_i should not have padding but got: {gt_i}"
        all_gt.append(gt_i)

        # update only valid decoder_hidden
        decoder_hidden[:, valid] = decoder_hidden_i
    
    all_decoder_outs = torch.cat(all_decoder_outs, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    # mean Negative Log Likelihood Loss
    loss = criterion(all_decoder_outs, all_gt)

    ## backpropagation
    loss.backward()

    ## update params
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def train(encoder, decoder, dataset, n_epochs, vocab, 
          batch_size=4, learning_rate=0.01, verbose=True, verbose_iter_interval=10):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate(vocab))
    total_iters = len(dataloader)
    epoch_losses = torch.zeros(size=[n_epochs], dtype=torch.double, device=DEVICE, requires_grad=False)
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(n_epochs):
        if verbose: print(f"Starting epoch {epoch+1}/{n_epochs}")
        epoch_loss = 0 # accumulate total loss during epoch
        print_epoch_loss = 0 # accumulate losses for printing
        for iter_idx, (ingredients, recipes, ing_lens, rec_lens) in enumerate(dataloader):
            if verbose and iter_idx > 0  and iter_idx % verbose_iter_interval == 0:
                print(f"(Epoch {epoch}, iter {iter_idx}/{total_iters}) Average loss so far: {print_epoch_loss/verbose_iter_interval:.3f}")
                print_epoch_loss = 0
            loss = train_iter(ingredients, recipes, ing_lens, rec_lens, encoder, decoder, 
                                   encoder_optimizer, decoder_optimizer, criterion, 
                                   vocab=vocab # remove later
                                   )
            epoch_loss += loss
            print_epoch_loss += loss
        epoch_loss /= total_iters # get average epoch loss
        if verbose: print(f"Average epoch loss: {epoch_loss:.3f}")
        epoch_losses.append(epoch_loss)

    return epoch_losses