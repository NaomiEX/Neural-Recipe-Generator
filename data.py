import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from constants import *

def replace(df, patterns, replacements, columns, regex=False):
    # patterns: List[str]
    # replacements: List[str]
    # columns: List[str]
    if type(replacements) == str:
        replacements = [replacements] * len(patterns)
    if type(columns) == str:
        columns = [columns] * len(patterns)

    for pat, rep, col in zip(patterns, replacements, columns):
        df[col] = df[col].str.replace(pat, rep, regex=regex)


def add_tags(df, add_intermediate_tag=False):
    assert 'Ingredients' in df.columns and 'Recipe' in df.columns

    replace(df, ['\t'], ' <INGREDIENT> ' if add_intermediate_tag else ' ', 'Ingredients')
    df.Ingredients = '<INGREDIENT_START> ' + df.Ingredients + ' <INGREDIENT_END>'

    replace(df, ['.', ';'], ' <RECIPE_STEP> ' if add_intermediate_tag else ' ', 'Recipe')
    df.Recipe = '<RECIPE_START> ' + df.Recipe + ' <RECIPE_END>'

def preprocess_data(orig_df, max_ingr_len=150, max_recipe_len=600, min_recipe_len=5, min_ingredients=1,
                    add_intermediate_tag=False):
    df = orig_df.copy() # ensure original data is not mutated (mostly for verification purposes)

    ## drop NA
    df = df.dropna()

    ## keep only rows with all lowercase (Recipe column is all lowercase already)
    df = df[df.Ingredients.str.islower()]

    ## replace brackets with space
    replace(df, ['[()]'], ' ', ['Ingredients', 'Recipe'], regex=True)

    ## add spaces around non-words (exclude whitespace, apostrophe, period (treated separately below))
    replace(df, ["([^0-9a-zA-Z.'\"/ ])"]*2, r" \1 ", ['Ingredients', 'Recipe'], regex=True)
    # add spaces around periods (excluding decimal places)
    replace(df, [r"\.(?!\d)"]*2, r" . ", ['Ingredients', 'Recipe'], regex=True)
    # add spaces around word/word
    replace(df, [r"([^0-9])\/([^0-9])"]*2, r"\1 / \2", ['Ingredients', 'Recipe'], regex=True)

    ## add tags for ingredients and recipes
    add_tags(df, add_intermediate_tag=add_intermediate_tag)

    ## replace >1 whitespace with a single space
    replace(df, ['[ ]{2,}']*2, " ", ['Ingredients', 'Recipe'], regex=True)

    ## remove leading and trailing whitespace
    df.Ingredients = df.Ingredients.str.strip()
    df.Recipe = df.Recipe.str.strip()

    if add_intermediate_tag:
        ## remove consecutive tags, for ex. <INGREDIENT>[0 or more whitespace]<INGREDIENT>
        replace(df, ["<INGREDIENT>[ \t\n]*([ \t\n]*<INGREDIENT>)+", "<RECIPE_STEP>[ \t\n]*([ \t\n]*<RECIPE_STEP>)+"], 
        ["<INGREDIENT>", "<RECIPE_STEP>"], ["Ingredients", "Recipe"], regex=True)

    ## filter out recipes and ingredients above/below limit
    recipe_lens = df.Recipe.apply(lambda r: len(r.split()))
    df = df[(recipe_lens > min_recipe_len) & (recipe_lens < max_recipe_len)]
    df = df[df.Ingredients.apply(lambda i: len(i.split())) < max_ingr_len]

    if add_intermediate_tag:
        ## filter out those with <1 ingredients
        df = df[df.Ingredients.str.count('<INGREDIENT>') >= min_ingredients]

    df = df.reset_index(drop=True)

    print(f"Number of data samples before preprocessing: {len(orig_df)}\n"
          f"Number of data samples after preprocessing: {len(df)} ({len(df) * 100/len(orig_df):.3f}%)")

    return df

class Vocabulary:
    def __init__(self, add_intermediate_tag=False):
        """Vocabulary class which can convert a valid word to unique index and converting the index back to word."""
        special_tags = dict(SPECIAL_TAGS)
        if not add_intermediate_tag:
            special_tags.pop(ING)
            special_tags.pop(REC)
        ## initialize
        self._word2index = special_tags
        self.word2count = {k: 0 for k in special_tags.keys()}
        self.index2word = {v:k for k,v in special_tags.items()}
        self.n_unique_words = len(self.index2word) # total number of words in the dictionary.

    def __len__(self):
        return len(self._word2index)

    def add_word(self, word):
        if word not in self._word2index:
            self._word2index[word] = self.n_unique_words
            self.index2word[self.n_unique_words] = word
            self.n_unique_words += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def word2index(self, word):
        if word not in self._word2index:
            return self._word2index[UNKNOWN_WORD]
        return self._word2index[word]

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_padding(self):
        # NOTE: should be called after finished with building vocab
        self.add_word(PAD_WORD)

    def add_unknown(self):
        self.add_word(UNKNOWN_WORD)

    def populate(self, df):
        for rowid in tqdm(range(len(df))):
            df_row = df.iloc[rowid]
            for i in range(2):
                self.add_sentence(df_row.iloc[i])
        self.add_unknown() # unknown word is for words in the dev/test not present in train
        self.add_padding() # padding should be last in the vocabulary (for convenience in decoder)

class RecipeDataset(Dataset):
    def __init__(self, df, vocab, train=True):
        """
        Args:
            df (pd.DataFrame): dataframe with two columns: "Ingredients" and "Recipe"
            vocab (Vocabulary): to convert word2index
        """
        super().__init__()
        self.ingredient_recipe_df = df
        self.vocab = vocab
        self.train = train

    def __len__(self):
        return len(self.ingredient_recipe_df)
    
    def __getitem__(self, index):
        row = self.ingredient_recipe_df.iloc[index]
        ingredient_tens = torch.tensor([self.vocab.word2index(w) for w in row.Ingredients.split(" ")],
                                       dtype=torch.long, device=DEVICE)
        if self.train:
            recipe_tens = torch.tensor([self.vocab.word2index(w) for w in row.Recipe.split(" ")],
                                        dtype=torch.long, device=DEVICE)
        else:
            recipe_tens = row.Recipe.split(" ") # List[str]
        return (ingredient_tens, recipe_tens)
    
# inspired by https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
def pad_collate(vocab, train=True):

    def _pad_collate(batch):
        # print(len(batch))
        # print(batch[0])
        # ingredients: tuple of len batch_size with Tensor elements containing all ingredients in batch
        # recipes: tuple of len batch_size with Tensor elements containing all recipes in batch
        #           (or in eval:) tuple of len batch_size with elements List[str]
        # print(batch)
        ingredients, recipes = zip(*batch)
        # print(ingredients)
        # print(recipes)
        ingr_lens = torch.tensor([len(x) for x in ingredients], dtype=torch.long, device=DEVICE)
        ingredients_padded = pad_sequence(ingredients, batch_first=True, padding_value=vocab.word2index(PAD_WORD))

        if train:
            recipe_lens = torch.tensor([len(r) for r in recipes], dtype=torch.long, device=DEVICE)
            recipes_padded = pad_sequence(recipes, batch_first=True, padding_value=vocab.word2index(PAD_WORD))
        else:
            recipe_lens = None
            recipes_padded = list(recipes)

        return ingredients_padded, recipes_padded, ingr_lens, recipe_lens
    
    return _pad_collate

def pack(x_embed, x_lens):
    # convert tensor with padding to a PackedSequence, this allows rnns to ignore paddings
    return pack_padded_sequence(x_embed, x_lens.cpu().int(), batch_first=True, enforce_sorted=False)

def unpack(out_packed, padding_val):
    out_padded, out_lens = pad_packed_sequence(out_packed, batch_first=True, padding_value=padding_val)
    return out_padded, out_lens