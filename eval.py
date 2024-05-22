import re
import torch
import json
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from tqdm import tqdm

from data import pad_collate

from constants import *

def get_ingredients_regex(ingredients_lst):
    return r'\b(?:' + '|'.join(re.escape(i) for i in ingredients_lst) + r')\b'

def get_all_ingredients(fpath, reverse_sort=True):
    # get list of valid ingredients in the dataset from RecipeWithPlans
    with open(fpath, "r") as f:
        all_ings = json.load(f)
    if reverse_sort:
        # sort by longest to shortest ingredients
        all_ings = sorted(all_ings, key=lambda i: len(i.split()), reverse=True)
    return all_ings

def find_ingredients_in_text(txt, regex, enforce_unique=True):
    res = re.findall(regex, txt)
    if enforce_unique:
        res = set(res)
    return res

def calc_bleu(gt_recipes, gen_recipes, split_gt=False, split_gen=False):
    """Calculate corpus BLEU-4 score.

    Args:
        gt_recipes (List): len N
        gen_recipes (List[List] or List): 
    """
    gt_recipes_lst2 = [[gt.split()] if split_gt else [gt] for gt in gt_recipes]
    if split_gen:
        gen_recipes = [r.split() for r in gen_recipes]
    return corpus_bleu(gt_recipes_lst2, gen_recipes)

def calc_meteor(gt_recipes, gen_recipes, split_gt=False, split_gen=False):
    if split_gt:
        gt_recipes_lst2 = [gt.split() for gt in gt_recipes]
    else:
        gt_recipes_lst2 = gt_recipes

    meteor_score = 0
    for i in tqdm(range(len(gen_recipes))):
        generated_recipe = gen_recipes[i].split() if split_gen else gen_recipes[i]
        gt_recipes_i = gt_recipes_lst2[i]
        meteor_score += meteor([gt_recipes_i], generated_recipe)
    return meteor_score / len(gen_recipes)

def get_prop_input_num_extra_ingredients(ingr_txt, recipe_txt, all_ings_regex, verbose=False, metric_sample=False):
    """get proportion of input ingredients and number of extra ingredients in recipe (`txt`).

    Args:
        ingr_txt (str): ingredients text
        recipe_txt (str): recipe text
        all_ings_regex (str): regex to match all valid ingredients
        input_ingredients_iter (List or Set): iterable of unique ingredients list
    """
    # get set of input ingredients
    input_ingredients = find_ingredients_in_text(ingr_txt, all_ings_regex)
    # # get regex to match only input ingredients
    # input_ingredients_regex = get_ingredients_regex(input_ingredients)

    if metric_sample:
        all_ings_in_text = [i.strip('<>') for i in set(re.findall("<[A-Za-z ]+>", recipe_txt))]
        input_ings_in_text = []
        extra_ings_in_text = []
        for ing in all_ings_in_text:
            if ing in input_ingredients:
                input_ings_in_text.append(ing)
            else:
                extra_ings_in_text.append(ing)
    else:
        # gets all ingredients in recipe
        all_ings_in_text = find_ingredients_in_text(recipe_txt, all_ings_regex)
        # gets only input ingredients in recipe
        # input_ings_in_text = find_ingredients_in_text(recipe_txt, input_ingredients_regex)
        input_ings_in_text = [i for i in all_ings_in_text if i in input_ingredients]
        extra_ings_in_text = [i for i in all_ings_in_text if i not in input_ings_in_text]

    if verbose:
        print(f"=====Input ingredients in text=====\n{input_ings_in_text}")
        print(f"\n=====All ingredients in text===== \n{all_ings_in_text}")

    prop_input_ings = len(input_ings_in_text) / len(input_ingredients)
    num_extra_ings = len(extra_ings_in_text)
    return prop_input_ings, num_extra_ings

## Gold comparison
def load_metric_sample(fpath):
    with open(fpath, "r") as f:
        metric_sample = f.readlines()

    metric_sample_ings = None
    metric_sample_gold_recipe = None
    metric_sample_generated_recipe = None

    for i in range(len(metric_sample)-1):
        l = metric_sample[i].strip().lower()
        next_l = metric_sample[i+1].strip().lower()
        if l == "ingredients:":
            metric_sample_ings = next_l
        elif l == "gold recipe:":
            metric_sample_gold_recipe = next_l
        elif l == "generated recipe:":
            metric_sample_generated_recipe = next_l

    return metric_sample_ings, metric_sample_gold_recipe, metric_sample_generated_recipe


def eval_decoder_iter(decoder, decoder_hidden, decoder_cell, encoder_houts,
                      ingredients, max_recipe_len, vocab, decoder_mode="basic"):
    assert decoder_mode in ["basic", "attention"]
    
    N = ingredients.size(0)
    all_decoder_outs = [[REC_START] for _ in range(N)] # stores the decoder outputs for each batch sample

    valid = torch.ones([N], device=DEVICE).bool() # Tensor[N] 
    decoder_input = torch.full([N], SPECIAL_TAGS[REC_START], dtype=torch.long, device=DEVICE)
    
    for _ in range(max_recipe_len-1): # generations are bounded by max length (-1 because of EOS)
        decoder_hidden_i = decoder_hidden[:, valid] # [1, N_valid, H]
        decoder_cell_i = decoder_cell[:, valid]

        if decoder_mode == "basic":
            # decoder_out: log probabilities over vocab; [N_valid, |Vocab|-1]
            # decoder_hfinal: final hidden state; [num_layers=1, N_valid, H]
            decoder_out, decoder_hidden_i, decoder_cell_i = decoder(decoder_input, decoder_hidden_i, decoder_cell_i)
        elif decoder_mode == "attention":
            encoder_houts_i = encoder_houts[valid] # [N_valid, L_i, H]
            decoder_out, decoder_hidden_i, decoder_cell_i, attn_weights_i = decoder(
                decoder_input, decoder_hidden_i, decoder_cell_i, encoder_houts_i, ingredients[valid])
            
        # decoder_tok_preds: token with highest log probability
        decoder_topk_preds = decoder_out.topk(1)[1].reshape(-1) # [N_valid]

        ## store generated output
        for dec_idx, valid_n in zip(range(len(decoder_topk_preds)), valid.nonzero()):
            valid_idx = valid_n.item()
            all_decoder_outs[valid_idx].append(
                vocab.index2word[decoder_topk_preds[dec_idx].item()]) # str

        ## check for end of recipe
        not_eor= decoder_topk_preds != SPECIAL_TAGS[REC_END] # [N_valid]
        # update valid
        valid_temp = valid.clone() # to avoid single mem location error
        valid_temp[valid] = not_eor
        valid = valid_temp
        del valid_temp
        # valid = torch.logical_and(valid, not_eor)

        # update decoder input for next iteration
        decoder_input = decoder_topk_preds[not_eor] # [N_valid_next]

        # update only valid decoder_hidden
        decoder_hidden[:, valid] = decoder_hidden_i[:, not_eor]
        decoder_cell[:, valid] = decoder_cell_i[:, not_eor]

        if valid.sum() < 1:
            break
    else: # if did not break meaning 1 or more exceeded max generation limit
        # forcably insert recipe stop tok at the end
        for valid_n in valid.nonzero():
            valid_idx = valid_n.item()
            all_decoder_outs[valid_idx].append(REC_END)

    return all_decoder_outs

def get_predictions_iter(ingredients, ing_lens, encoder, decoder, vocab, max_recipe_len=600, 
                         decoder_mode="basic"):
    """Get predictions from trained model for a single iteration. Processes batched data.
    NOTE: ensure that this function is wrapped in `with torch.no_grad():`

    Args:
        ingredients (torch.Tensor): padded ingredients tensor in idx form; 
                                    shape [N, L_i], where L_i = max ingredients length in batch
        ing_lens (torch.Tensor): unpadded length of ingredients; shape [N]
        rec_lens (torch.Tensor): unpadded length of recipes; shape [N]
        encoder (EncoderRNN): encoder RNN module
        decoder (DecoderRNN): decoder RNN module
    """
    assert encoder.training is False and decoder.training is False

    N = ingredients.size(0)

    ## feed ingredients through encoder
    # enc_out: padded encoder output tensor with shape [N, L, H]
    # enc_out_lens: unpadded sequence lengths; tensor with shape [N]
    # enc_h_final: final hidden state: [num_layers=1, N, H]
    # enc_c_final: final cell state: [num_layers=1, N, H]
    enc_out, enc_out_lens, enc_h_final, enc_c_final = encoder(ingredients, ing_lens)
    
    # initialize decoder hidden state as final encoder hidden state
    decoder_hidden = enc_h_final
    decoder_cell = enc_c_final

    # List[List[str]]
    
    all_decoder_outs = eval_decoder_iter(decoder, decoder_hidden, decoder_cell,
                                         enc_out, ingredients, max_recipe_len,
                                         vocab, decoder_mode=decoder_mode)

    return all_decoder_outs

def eval(encoder, decoder, dataset, vocab, batch_size=4, max_recipe_len=600, decoder_mode="basic"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate(vocab, train=False))

    all_decoder_outs = [] # (List[List[str]]): List of len `N`, each element is the generated sequence for that sample
    all_gt_recipes = [] # (List[List[str]])

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for ingredients, recipes, ing_lens, _ in tqdm(dataloader):
            # ingredients: Tensor[N, L_i] padded ingredients
            # recipes (List[List[str]]): list of len N, each element is a
            #                               list of len `|gt_i|`, which is the length of the i-th ground-truth sequence

            # dec_outs (List[List[str]]): list of len `batch_size`, each element is a 
            #                               list of len `gen_size`, which is the size of the generated sequence, and each element is a 
            #                                   str representing a single word in the generated recipe
            dec_outs = get_predictions_iter(ingredients, ing_lens, encoder, decoder, vocab, 
                                            max_recipe_len=max_recipe_len, decoder_mode=decoder_mode)
            
            all_decoder_outs += dec_outs
            all_gt_recipes += recipes

    # TODO: INTEGRATE CALCULATION OF METRICS IN HERE ONCE STABLE
    
    return all_decoder_outs, all_gt_recipes
