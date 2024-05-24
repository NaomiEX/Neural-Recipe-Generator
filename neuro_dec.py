import torch
from copy import deepcopy

from constants import *
from data import *
from eval import *


def repeat_for_k(x, k, dim=1):
    dims = [1]*x.ndim
    dims[dim] = k
    return x.repeat(dims)

def detect_negative_constraint(num_ks, curr_decoder_outs_shape, all_decoder_outs, neg_constraints):
    """_summary_

    Args:
        num_ks (int): number of hypothesis which have not ended
        curr_decoder_outs_shape: shape of current decoder output [k, |V|-1]
        all_decoder_outs (list): [k, |generation|]
        neg_constraints (List[List[int]]): list of negative constraints, nested list because constraints can be multi-word
    """

    # >1 if neg constraint satisfied, 0 otherwise
    # >1 makes sense if multiple negative constraints are satisfied (yielding larger penalty)
    neg_constraint_satisfied = torch.zeros(curr_decoder_outs_shape, device=DEVICE) # [k, |V|-1]

    for ki in range(num_ks):
        for neg_cons in neg_constraints:
            neg_constraint_exists = True
            neg_idx = neg_cons[-1]

            if (len(neg_cons) - 1) > len(all_decoder_outs[ki]):
                continue # i.e. neg_constraint_exists=False

            # to check if a negative constraint is satisfied (irreversible unsatisfaction) 
            # so need to check from back to front (only relevant for multi-word constraints)
            for word_idx, constraint_word in enumerate(neg_cons[:-1][::-1]):
                # if mismatch, then neg constraint is not satisfied
                if all_decoder_outs[ki][-(word_idx+1)] != constraint_word:
                    neg_constraint_exists = False
                    break

            neg_constraint_satisfied[ki][neg_idx] += neg_constraint_exists
    return neg_constraint_satisfied

def detect_low_likelihood(alpha, likelihood):
    """Detect likelihoods < top-alpha

    Args:
        alpha (_type_): _description_
        likelihood (_type_): [k, |V|-1]
    """
    # get the minimum value to be included within the top-alpha
    likelihood_penalty_thresh = likelihood.flatten().topk(alpha).values.min()

    return likelihood < likelihood_penalty_thresh

def update_irreversible_satisfaction(num_ks, k_irreversible_satisfaction, all_decoder_outs, pos_constraints,
                                   out_size):
    """_summary_

    Args:
        num_ks (int): number of hypothesis which have not ended
        beta (_type_): top-beta number of irreversible satisfactions to keep
        k_irreversible_satisfaction (_type_): number of irreversible satisfactions per hypothesis; [k]
        all_decoder_outs (_type_): list of previous generations for each hypothesis [k, |generation|]
        pos_constraints (List[List[List[int]]]): 3D list of shape [max_k, num positive constraints, length of positive constraint]
        out_size: dimension of generations |Vocab|-1
    """
    # [k, |V|-1]
    k_irreversible_satisfaction_now = k_irreversible_satisfaction[:, None].repeat(1, out_size)
    pos_constraints_satisfied = torch.full_like(k_irreversible_satisfaction_now, -1)

    for ki in range(num_ks):
        for pos_cons_idx, pos_cons in enumerate(pos_constraints[ki]):
            pos_constraint_exist = True
            
            ## similar to detecting irreversible unsatisfaction, 
            # we check from last (current) to first (previous generated text)
            pos_idx = pos_cons[-1]

            if (len(pos_cons) - 1) > len(all_decoder_outs[ki]):
                continue # i.e. pos_constraint_exist=False

            # checkds from 2nd last to first (only relevant for multi-word constraints)
            for word_idx, constraint_word in enumerate(pos_cons[:-1][::-1]): 
                if all_decoder_outs[ki][-(word_idx+1)] != constraint_word:
                    pos_constraint_exist = False
                    break

            if pos_constraint_exist:
                k_irreversible_satisfaction_now[ki][pos_idx] += 1
                pos_constraints_satisfied[ki][pos_idx] = pos_cons_idx
                
    return k_irreversible_satisfaction_now, pos_constraints_satisfied

def detect_low_irreversible_satisfactions(k_irreversible_satisfactions_now, # [k, |V|-1]
                                          beta):
    # get the minimum number of satisfied clauses to be included within the top-beta
    # need to use unique because many candidates can have the same number of satisfied clauses
    unique_num_irreversible_satisfactions = k_irreversible_satisfactions_now.flatten().unique()
    satisfaction_penalty_thresh = unique_num_irreversible_satisfactions[-min(beta, len(unique_num_irreversible_satisfactions))].item()
    return k_irreversible_satisfactions_now < satisfaction_penalty_thresh

def get_proportion_completion_reward(num_ks,
                                     scores, # [k, |V|-1]
                                     pos_constraints, # List[List[List[int]]]; [k, num constraints, len constraint]
                                     all_decoder_outs, # List [k, |generation|]
                                     lam=0.75
                                     ):
    reward = torch.zeros_like(scores)

    for ki in range(num_ks):
        for pos_cons in pos_constraints[ki]:
            ## just like in the paper, we also reward partial completion (reversible satisfaction)
            # to do this we need to do constraint prefix comparison for lengths i=0...|constraint|
            # because if a constraint is: [0, 1, 2, 3], a partial completion could be [0], [0,1], [0,1,2]
            # with full completion: [0,1,2,3]
            for word_idx, constraint_word in enumerate(pos_cons):
                if word_idx > len(all_decoder_outs[ki]):
                    break
                if word_idx == 0 or all_decoder_outs[ki][-word_idx:] == pos_cons[:word_idx]:
                    reward[ki][constraint_word] = max((word_idx+1) / len(pos_cons), reward[ki][constraint_word])
    
    return lam * reward

def neurologic_decoding(decoder, decoder_hidden, decoder_cell, encoder_houts,
                            ingredients, max_recipe_len,
                            pos_constraints, neg_constraints, k, alpha, beta,
                            neg_constraint_penalty, likelihood_penalty, low_irr_satisfaction_penalty, lam,
                            decoder_mode="attention"):
    """Neurological decoding for a particular sample in batch.

    Args:
        decoder (_type_): _description_
        decoder_hidden (_type_): [1, N=1, H]
        decoder_cell (_type_): [1, N=1, H]
        encoder_houts (_type_): [N=1, L_i, H]
        ingredients (_type_): [N=1, L_i]
        max_recipe_len (_type_): _description_
        pos_constraints (List[List[int]]): list of positive constraints, nested list because constraints can be multi-word 
                                           IMPORTANT: these are expected to be transformed to index using vocab
        neg_constraints (List[List[int]]): list of negative constraints, nested list because constraints can be multi-word
                                           IMPORTANT: these are expected to be transformed to index using vocab
        k (_type_): number of hypothesis per sample
        alpha (_type_): top-alpha likelihood which are not pruned
        beta (_type_): top-beta number of satisfied clauses which are not pruned
        neg_constraint_penalty (_type_): penalty for including negative constraint
        likelihood_penalty (_type_): penalty for not being in top-alpha likelihood
        low_irr_satisfaction_penalty (_type_): penalty for not being in top-beta no. of satisfied clauses
        lam (_type_): lambda to add constraint progress to score
        decoder_mode (str, optional): _description_. Defaults to "basic".
    """
    assert decoder_mode == "attention", "best model is attention, should be using attention!"

    K = torch.tensor([0]).to(DEVICE) # start with 1 hypothesis
    
    all_decoder_outs = [[SPECIAL_TAGS[REC_START]] for _ in range(k)] # stores decoder outputs for each hypothesis; [max_K]

    decoder_input = torch.full([1], SPECIAL_TAGS[REC_START], dtype=torch.long, device=DEVICE)

    # stores the running likelihoods for the k hypotheses
    k_likelihoods = torch.zeros([k], dtype=torch.float, device=DEVICE) # [max_K]

    # number of irreversible satisfactions so far for each hypothesis
    k_irreversible_satisfaction = torch.zeros_like(decoder_input) # [k]

    # lists *remaining* positive constraints for each hypotheses
    # once a positive constraint has been fully satisfied (irreversible satisfaction), it is removed
    # 3D list of shape [max_k, num positive constraints, length of positive constraint]
    # pos_constraints_i = [pos_constraints for _ in range(k)]
    pos_constraints_i = [pos_constraints]

    ## initialize inputs as the same for all ks because all of them have the same ingredients
    # encoder_houts_i = repeat_for_k(encoder_houts, k, dim=0) # [N=max_K, L_i, H]
    # decoder_hidden_i = repeat_for_k(decoder_hidden, k, dim=1) # [1, N=max_K, H]
    # decoder_cell_i = repeat_for_k(decoder_cell, k, dim=1) # [1, N=max_K, H]
    # ingredients_i = repeat_for_k(ingredients, k, dim=0) # [N=max_K, L_i]
    encoder_houts_i = encoder_houts
    decoder_hidden_i = decoder_hidden
    decoder_cell_i = decoder_cell
    ingredients_i = ingredients

    for recipe_i in range(max_recipe_len - 1): # generations are bounded by max length (-1 because of EOS)
        ## precondition: K is the list of hypotheses which have not ended

        num_ks = len(K) # some hypotheses can finish early so need to udpate this every iter
        valid_all_decoder_outs = [all_decoder_outs[i] for i in K]

        ## attention
        # decoder_out: log probs [k, |Vocab|-1]
        decoder_out, decoder_hidden_i, decoder_cell_i, attn_weights_i = decoder(
            decoder_input, decoder_hidden_i, decoder_cell_i, encoder_houts_i, ingredients_i
        )

        # sum all log probs with running log probs
        # [k, |V|-1] + [k, 1] = [k, |V|-1]
        likelihood_i = decoder_out + k_likelihoods[K].unsqueeze(-1)

        scores = decoder_out.clone() # used for selection (can no longer be interpreted as probabilities so we preserve likelihoods)

        ############# PRUNING #############

        # detect generations which will cause irreversible unsatisfaction; [k, |V|-1]
        neg_constraint_satisfied = detect_negative_constraint(
            num_ks, likelihood_i.shape, valid_all_decoder_outs, neg_constraints)
        
        # detect generations with low likelihood; [k, |V|-1]
        low_likelihoods = detect_low_likelihood(alpha, likelihood_i)

        # get potential total irreversible satisfaction (including already satisfied clauses) for each candidate
        # k_irreversible_satisfaction_now: [k, |V|-1]
        # pos_constraints_satisfied: [k, |V|-1]
        k_irreversible_satisfaction_now, pos_constraints_satisfied = update_irreversible_satisfaction(
            num_ks, k_irreversible_satisfaction, valid_all_decoder_outs, pos_constraints_i, 
            out_size=likelihood_i.size(-1))
        
        # detect generations with < top-beta number of irreversibly satisfied clauses; [k, |V|-1]
        low_irreversible_satisfaction = detect_low_irreversible_satisfactions(
            k_irreversible_satisfaction_now, beta)
        
        # perform soft pruning, i.e. penalizing instead of filtering out (see report)
        penalties = neg_constraint_satisfied * neg_constraint_penalty + \
                    low_likelihoods * likelihood_penalty + \
                    low_irreversible_satisfaction * low_irr_satisfaction_penalty
        
        scores -= penalties
        
        ############# SELECTION #############

        # get rewards for partial/full completion
        rewards = get_proportion_completion_reward(num_ks, scores, pos_constraints_i, valid_all_decoder_outs, lam=lam)

        scores += rewards # [k, |V|-1]

        # select top-k based on scores across all candidates
        topk_scores, topk_inds = scores.flatten().topk(num_ks if recipe_i > 0 else k)
        k_origin = torch.div(topk_inds, scores.size(-1), rounding_mode="floor")
        k_origin_global = K[k_origin] # get global k
        word_idx = topk_inds % scores.size(-1) # k top words

        prev_all_decoder_outs = deepcopy(all_decoder_outs)
        for wi, (ki, k_glob) in enumerate(zip((K if recipe_i > 0 else range(3)), k_origin_global)):
            all_decoder_outs[ki] = prev_all_decoder_outs[k_glob] + [word_idx[wi].item()]
            # all_decoder_outs[k_glob].append(word_idx[k_glob].item())

        k_likelihoods[K if recipe_i > 0 else range(k)] = likelihood_i[k_origin, word_idx] # [k]

        pos_constraints_chosen = [pos_constraints_i[ki] for ki in k_origin]

        for i, (ki, wi) in enumerate(zip(k_origin, word_idx)):
            # remove irreversibly satisfied constraint
            pos_constraints_chosen[i] = [c for cidx, c in enumerate(pos_constraints_chosen[i])
                                         if cidx != pos_constraints_satisfied[ki, wi]]
        pos_constraints_i = pos_constraints_chosen

        ############# PREPARE FOR NEXT ITERATION #############

        ## check if any of the hypotheses has ended (update K)
        not_eor = word_idx != SPECIAL_TAGS[REC_END]
        K = torch.arange(k, device=DEVICE)[K][not_eor] if recipe_i > 0 \
            else torch.arange(k, device=DEVICE)[not_eor]
        # K=K[not_eor] if recipe_i > 0 else torch.arange(k)[not_eor]
        # K=torch.arange(len(K) if recipe_i > 0 else k)[not_eor]

        if len(K) < 1:
            break

        ## postcondition: K is the list of hypotheses which have not ended

        ## determine inputs for next iter

        k_irreversible_satisfaction = k_irreversible_satisfaction_now[k_origin[not_eor], word_idx[not_eor]] # [k]
        decoder_input = word_idx[not_eor] # [k]

        ## all the same so no need to index using K
        if recipe_i == 0:
            encoder_houts_i = repeat_for_k(encoder_houts, len(K), dim=0)
            ingredients_i = repeat_for_k(ingredients, len(K), dim=0)
        else:
            encoder_houts_i = encoder_houts_i[:len(K)] # [N=k, L_i, H]
            ingredients_i = ingredients_i[:len(K)]

        ## different based on k
        decoder_hidden_i = decoder_hidden_i[:, k_origin[not_eor]]
        decoder_cell_i = decoder_cell_i[:, k_origin[not_eor]]
        pos_constraints_i = [pos_constraints_i[i] for i in not_eor.nonzero().flatten()]
    else:
        for ki in K.tolist():
            all_decoder_outs[ki].append(SPECIAL_TAGS[REC_END])

    k_likelihoods_normalized = k_likelihoods / torch.tensor([len(o) for o in all_decoder_outs],
                                                            device=k_likelihoods.device)

    return all_decoder_outs, k_likelihoods_normalized

def eval_neuro_decoding_iter(ingredients, ing_lens, encoder, decoder, vocab, pos_constraints, neg_constraints,
                              max_recipe_len=MAX_RECIPE_LEN,
                              **kwargs):
    assert encoder.training is False and decoder.training is False

    enc_out, enc_out_lens, enc_h_final, enc_c_final = encoder(ingredients, ing_lens)
    
    # initialize decoder hidden state as final encoder hidden state
    decoder_hidden = enc_h_final
    decoder_cell = enc_c_final

    # all_decoder_outs (List[List[int]]): list of k hypotheses
    # k_likelihoods_normalized (tensor): has overall likelihood of each hypotheses shape [k]
    all_decoder_outs, k_likelihoods_normalized = neurologic_decoding(decoder, decoder_hidden, decoder_cell, enc_out,
                                                                     ingredients, max_recipe_len, pos_constraints,
                                                                     neg_constraints, **kwargs)
    
    max_i = k_likelihoods_normalized.argmax()
    final_decoder_out = all_decoder_outs[max_i]

    final_decoder_out_txt = [vocab.index2word[w] for w in final_decoder_out]

    return final_decoder_out_txt

def eval_neuro_decoding(encoder, decoder, dataset, vocab, all_ingredients_list, 
                        max_recipe_len=MAX_RECIPE_LEN, **kwargs):
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                              collate_fn=pad_collate(vocab, train=False))
    
    all_decoder_outs = [] # (List[List[str]]): List of len `N`, each element is the generated sequence for that sample
    all_gt_recipes = [] # (List[List[str]])
    all_ingredients = []

    ## Build constraints dictionary (all possible constraints)
    constraints_dict = {}
    valid_ingredients_list = deepcopy(all_ingredients_list)
    invalid_ingredients = []
    for ingredient in valid_ingredients_list:
        invalid_ingredient=False
        ingredient_idx_form = []

        for word in ingredient.split():
            if not vocab.word_exist_in_vocab(word):
                invalid_ingredient = True
                break
            ingredient_idx_form.append(vocab.word2index(word))
        if not invalid_ingredient:
            constraints_dict[ingredient] = ingredient_idx_form
        else:
            # valid_ingredients_list.remove(ingredient)
            invalid_ingredients.append(ingredient)

    valid_ingredients_list = [i for i in valid_ingredients_list if i not in invalid_ingredients]

    all_ingredients_regex = get_ingredients_regex(valid_ingredients_list)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for ingredients, recipes, ing_lens, _ in tqdm(dataloader):
            pos_constraints, neg_constraints, ingredients_text = create_pos_neg_constraints(
                ingredients, vocab, all_ingredients_regex, constraints_dict)
            final_decoder_out_txt = eval_neuro_decoding_iter(
                ingredients, ing_lens, encoder, decoder, vocab, pos_constraints, neg_constraints,
                max_recipe_len, **kwargs
            )
            all_decoder_outs += [final_decoder_out_txt]
            all_gt_recipes += recipes
            all_ingredients += [[ingredients_text]]

    return all_decoder_outs, all_gt_recipes, all_ingredients

def create_pos_neg_constraints(ingredients_idxs, # expecting batch size 1
                               vocab,
                               all_ingredients_regex,
                               constraints_dict):
    
    ingredients_text = " ".join([vocab.index2word[i] for i in ingredients_idxs[0].tolist()])
    input_ingredients = find_ingredients_in_text(ingredients_text, all_ingredients_regex)
    input_ingrs_partial = ' '.join(input_ingredients).split()
    pos_constraints = [constraints_dict[ing] for ing in sorted(list(input_ingredients), key=len)]
    neg_constraints = []
    for constraint in constraints_dict.keys():
        if constraint in pos_constraints:
            continue
        valid_negative_constraint=True
        for word in constraint.split():
            if word in input_ingrs_partial:
                valid_negative_constraint = False
                break
        if valid_negative_constraint:
            neg_constraints.append(constraints_dict[constraint])
    return pos_constraints, neg_constraints, ingredients_text
