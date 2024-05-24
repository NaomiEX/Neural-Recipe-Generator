import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import pad_collate
from eval import calc_bleu, calc_meteor, eval
from utils import save_model

from constants import *

def train_decoder_iter(decoder, decoder_hidden, decoder_cell, encoder_houts, 
                       ingredients, recipes, padded_rec_len, rec_lens,
                       pad_word_idx, decoder_mode="basic"):
    assert decoder_mode in ["basic", "attention"]

    all_decoder_outs = [] # List of [N, |Vocab|-1]
    all_gt = [] # List of [N]
    ## NOTE: recipes already contain start token no need to add manually
    # encoder_houts_i = encoder_houts # [N, L_i, H]
    for di in range(padded_rec_len-1):
        # get batches which have valid (non-padding and non ending) tokens as input
        valid = (rec_lens - 1) > di
        decoder_input_i = recipes[valid, di] # [N_valid]
        decoder_hidden_i = decoder_hidden[:,valid] # [n_layers=1, N_valid, H]
        decoder_cell_i = decoder_cell[:, valid] # [n_layers=1, N_valid, H]

        if decoder_mode == "basic":
            # decoder_out: log probabilities over vocab; [N_valid, |Vocab|-1]
            # decoder_hfinal: final hidden state; [num_layers=1, N_valid, H]
            decoder_out, decoder_hidden_i, decoder_cell_i = decoder(
                decoder_input_i, decoder_hidden_i, decoder_cell_i)
            attn_weights = None
        elif decoder_mode == "attention":
            encoder_houts_i = encoder_houts[valid] # [N_valid, L_i, H]
            decoder_out, decoder_hidden_i, decoder_cell_i, attn_weights_i = decoder(
                decoder_input_i, decoder_hidden_i, decoder_cell_i, encoder_houts_i, ingredients[valid])

        all_decoder_outs.append(decoder_out)

        # because we ensured that input cannot be end token, there is a guaranteed non-padding token
        # for each valid batch sample
        gt_i = recipes[valid, di+1] # [N_valid]
        assert (gt_i != pad_word_idx).all(), f"gt_i should not have padding but got: {gt_i}"
        all_gt.append(gt_i)

        # update only valid decoder_hidden and decoder_cell
        decoder_hidden[:, valid] = decoder_hidden_i
        decoder_cell[:, valid] = decoder_cell_i

    return all_decoder_outs, all_gt

def train_iter(ingredients, recipes, ing_lens, rec_lens, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
               decoder_mode="basic", prevent_pretrained_grad_update=True, vocab=None # !remove later
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
    # enc_c_final: final cell state: [num_layers=1, N, H]
    enc_out, enc_out_lens, enc_h_final, enc_c_final = encoder(ingredients, ing_lens)

    # initialize decoder hidden state and cell state as final encoder hidden and cell state
    decoder_hidden = enc_h_final
    decoder_cell = enc_c_final

    if TEACHER_FORCING_RATIO < 1:
        raise ValueError("Non-teacher forcing is not implemented")
    
    loss = 0

    all_decoder_outs, all_gt = train_decoder_iter(decoder, decoder_hidden, decoder_cell, enc_out, 
                                                  ingredients, recipes, padded_rec_len, rec_lens, 
                                                  vocab.word2index(PAD_WORD), decoder_mode=decoder_mode)
    
    all_decoder_outs = torch.cat(all_decoder_outs, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    # mean Negative Log Likelihood Loss
    loss = criterion(all_decoder_outs, all_gt)

    ## backpropagation
    loss.backward()

    # print("=====BEFORE MASKING=====")
    # print("ENCODER EMBEDDING WEIGHT GRADS of shape:", encoder.embedding.weight.grad.shape)
    # print(encoder.embedding.weight.grad[:10, :10])

    # print("DECODER EMBEDDING WEIGHT GRADS of shape:", decoder.embedding.weight.grad.shape)
    # print(decoder.embedding.weight.grad[:10, :10])

    # print("=====AFTER MASKING=====")
    if prevent_pretrained_grad_update:
        encoder.update_embedding_grad(encoder.embedding.weight.grad)
        decoder.update_embedding_grad(decoder.embedding.weight.grad)
    # print("ENCODER EMBEDDING WEIGHT GRADS of shape:", encoder.embedding.weight.grad.shape)
    # print(encoder.embedding.weight.grad[:10, :10])

    # print("DECODER EMBEDDING WEIGHT GRADS of shape:", decoder.embedding.weight.grad.shape)
    # print(decoder.embedding.weight.grad[:10, :10])


    ## update params
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()

def train(encoder, decoder, encoder_optimizer, decoder_optimizer, dataset, n_epochs, vocab,
          decoder_mode="basic", batch_size=4, enc_lr_scheduler=None, dec_lr_scheduler=None, 
          dev_ds_val_loss=None, dev_ds_val_met=None, identifier="", min_bleu_to_save=0.01,
          verbose=True, verbose_iter_interval=10):
    assert (enc_lr_scheduler is None and dec_lr_scheduler is None) or (enc_lr_scheduler is not None and dec_lr_scheduler is not None)
    
    evaluate = dev_ds_val_loss is not None and dev_ds_val_met is not None
    assert (not evaluate) or (evaluate and len(identifier) > 0)
    
    use_scheduler =  enc_lr_scheduler is not None and dec_lr_scheduler is not None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate(vocab))
    total_iters = len(dataloader)
    
    
    epoch_losses = torch.zeros(size=[n_epochs], dtype=torch.double, device=DEVICE, requires_grad=False)
    val_epoch_losses = torch.zeros(size=[n_epochs], dtype=torch.double, device=DEVICE, requires_grad=False)
    
    criterion = nn.NLLLoss()
    log = ""
    
    highest_bleu = 0

    for epoch in range(n_epochs):
        if verbose: print(f"Starting epoch {epoch+1}/{n_epochs}, "
                          f"enc lr scheduler: {enc_lr_scheduler.get_last_lr()}, dec lr scheduler: {dec_lr_scheduler.get_last_lr()}" \
                            if use_scheduler else "")
        epoch_loss = 0 # accumulate total loss during epoch
        print_epoch_loss = 0 # accumulate losses for printing

        start_epoch_time = time.time()
        for iter_idx, (ingredients, recipes, ing_lens, rec_lens) in enumerate(dataloader):
            if verbose and iter_idx > 0  and iter_idx % verbose_iter_interval == 0:
                msg = f"(Epoch {epoch}, iter {iter_idx}/{total_iters}) Average loss so far: {print_epoch_loss/verbose_iter_interval:.3f}"
                log += msg + "\n"
                print(msg)
                print_epoch_loss = 0
            loss = train_iter(ingredients, recipes, ing_lens, rec_lens, encoder, decoder, 
                                   encoder_optimizer, decoder_optimizer, criterion, 
                                   decoder_mode=decoder_mode,
                                   vocab=vocab # remove later
                                   )
            epoch_loss += loss
            print_epoch_loss += loss
        end_epoch_time = time.time()
        epoch_loss /= total_iters # get average epoch loss
        if verbose: 
            one_epoch_time_sec = end_epoch_time - start_epoch_time
            remaining_epochs = n_epochs - epoch - 1
            remaining_time = one_epoch_time_sec * remaining_epochs
            remaining_time_hours = remaining_time //3600
            remaining_time_mins = remaining_time % 3600 // 60
            msg = f"Average epoch loss: {epoch_loss:.3f}\n"\
                  f"This epoch took {one_epoch_time_sec / 60} mins. Time remaining: {remaining_time_hours} hrs {remaining_time_mins} mins."
            log += msg + "\n"
            print(msg)

        epoch_losses[epoch]=epoch_loss
        if use_scheduler:
            enc_lr_scheduler.step()
            dec_lr_scheduler.step()

        if evaluate:
            ## get validation loss
            val_loss = get_validation_loss(encoder, decoder, dev_ds_val_loss, vocab, batch_size=batch_size,
                                           decoder_mode=decoder_mode)
            val_epoch_losses[epoch] = val_loss
            
            if verbose:
                msg = f"validation loss: {val_loss}"
                log += msg + "\n"
                print(msg)

            ## get validation metrics
            all_decoder_outs, all_gt_recipes = eval(encoder, decoder, dev_ds_val_met, vocab,
                                                    batch_size=batch_size, max_recipe_len=MAX_RECIPE_LEN,
                                                    decoder_mode=decoder_mode)
            bleu = calc_bleu(all_gt_recipes, all_decoder_outs)
            meteor = calc_meteor(all_gt_recipes, all_decoder_outs, split_gt=False)

            if verbose:
                msg = f"BLEU score: {bleu}, METEOR score: {meteor}"
                log += msg + "\n"
                print(msg)
            if bleu > highest_bleu:
                highest_bleu = bleu
                if bleu > min_bleu_to_save:
                    save_model(encoder, decoder, f"{identifier}_ep_{epoch}")
            
            # set back to train mode because in eval they are set to eval mode
            encoder.train()
            decoder.train()


    return epoch_losses, val_epoch_losses, log


def get_validation_loss_iter(encoder, decoder, ingredients, recipes, ing_lens, rec_lens, vocab, criterion,
                             decoder_mode="basic"):
    assert not encoder.training and not decoder.training
    padded_rec_len = recipes.size(1) # L_r

    ## feed ingredients through encoder
    # enc_out: padded encoder output tensor with shape [N, L, H]
    # enc_out_lens: unpadded sequence lengths; tensor with shape [N]
    # enc_h_final: final hidden state: [num_layers=1, N, H]
    # enc_c_final: final cell state: [num_layers=1, N, H]
    enc_out, enc_out_lens, enc_h_final, enc_c_final = encoder(ingredients, ing_lens)

    # initialize decoder hidden state and cell state as final encoder hidden and cell state
    decoder_hidden = enc_h_final
    decoder_cell = enc_c_final

    if TEACHER_FORCING_RATIO < 1:
        raise ValueError("Non-teacher forcing is not implemented")
    
    loss = 0

    all_decoder_outs, all_gt = train_decoder_iter(decoder, decoder_hidden, decoder_cell, enc_out, 
                                                  ingredients, recipes, padded_rec_len, rec_lens, 
                                                  vocab.word2index(PAD_WORD), decoder_mode=decoder_mode)
    
    all_decoder_outs = torch.cat(all_decoder_outs, dim=0)
    all_gt = torch.cat(all_gt, dim=0)

    # mean Negative Log Likelihood Loss
    loss = criterion(all_decoder_outs, all_gt)

    return loss.item()


def get_validation_loss(encoder, decoder, dataset, vocab, batch_size=4, decoder_mode="basic"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate(vocab))

    encoder.eval()
    decoder.eval()

    criterion = nn.NLLLoss()

    total_loss = 0

    with torch.no_grad():
        for idx, (ingredients, recipes, ing_lens, rec_lens) in tqdm(enumerate(dataloader)):
            loss = get_validation_loss_iter(encoder, decoder, ingredients, recipes, ing_lens, rec_lens,
                                            vocab, criterion, decoder_mode=decoder_mode)
            total_loss += loss

    return total_loss/len(dataloader)