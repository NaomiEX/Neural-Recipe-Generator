import re
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor

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

def calc_bleu(gt_recipes, gen_recipes, split_gt=True, split_gen=False):
    """Calculate corpus BLEU-4 score.

    Args:
        gt_recipes (List): len N
        gen_recipes (List[List] or List): 
    """
    gt_recipes_lst2 = [[gt.split()] if split_gt else [[gt]] for gt in gt_recipes]
    if split_gen:
        gen_recipes = [r.split() for r in gen_recipes]
    return corpus_bleu(gt_recipes_lst2, gen_recipes)

def calc_meteor(gt_recipes, gen_recipes, split_gt=True, split_gen=False):
    gt_recipes_lst2 = [gt.split() if split_gt else [gt] for gt in gt_recipes]

    meteor_score = 0
    for i in range(len(gen_recipes)):
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

