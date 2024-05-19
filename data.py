import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


## constants
SPECIAL_TAGS = {
    "<INGREDIENT_START>": 0,
    "<INGREDIENT>": 1,
    "<INGREDIENT_END>": 2,
    "<RECIPE_START>": 3,
    "<RECIPE_STEP>": 4,
    "<RECIPE_END>": 5
}

PAD_WORD = "<PAD>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##

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


def add_tags(df):
    assert 'Ingredients' in df.columns and 'Recipe' in df.columns

    replace(df, ['\t'], ' <INGREDIENT> ', 'Ingredients')
    df.Ingredients = '<INGREDIENT_START> ' + df.Ingredients + ' <INGREDIENT_END>'

    replace(df, ['.', ';'], ' <RECIPE_STEP> ', 'Recipe')
    df.Recipe = '<RECIPE_START> ' + df.Recipe + ' <RECIPE_END>'

def preprocess_data(orig_df, max_ingr_len=150, max_recipe_len=600, min_recipe_len=5, min_ingredients=1):
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

    ## add tags for ingredients and recipes
    add_tags(df)

    ## replace >1 whitespace with a single space
    replace(df, ['[ ]{2,}']*2, " ", ['Ingredients', 'Recipe'], regex=True)

    ## remove leading and trailing whitespace
    df.Ingredients = df.Ingredients.str.strip()
    df.Recipe = df.Recipe.str.strip()

    ## remove consecutive tags, for ex. <INGREDIENT>[0 or more whitespace]<INGREDIENT>
    replace(df, ["<INGREDIENT>[ \t\n]*([ \t\n]*<INGREDIENT>)+", "<RECIPE_STEP>[ \t\n]*([ \t\n]*<RECIPE_STEP>)+"], 
    ["<INGREDIENT>", "<RECIPE_STEP>"], ["Ingredients", "Recipe"], regex=True)

    ## filter out recipes and ingredients above/below limit
    recipe_lens = df.Recipe.apply(lambda r: len(r.split()))
    df = df[(recipe_lens > min_recipe_len) & (recipe_lens < max_recipe_len)]
    df = df[df.Ingredients.apply(lambda i: len(i.split())) < max_ingr_len]

    ## filter out those with <1 ingredients
    df = df[df.Ingredients.str.count('<INGREDIENT>') >= min_ingredients]

    print(f"Number of data samples before preprocessing: {len(orig_df)}\n"
          f"Number of data samples after preprocessing: {len(df)} ({len(df) * 100/len(orig_df):.3f}%)")

    return df

class Vocabulary:
    def __init__(self):
        """Vocabulary class which can convert a valid word to unique index and converting the index back to word."""
        ## initialize
        self.word2index = SPECIAL_TAGS
        self.word2count = {k: 0 for k in SPECIAL_TAGS.keys()}
        self.index2word = {v:k for k,v in SPECIAL_TAGS.items()}
        self.n_unique_words = len(self.index2word) # total number of words in the dictionary.

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_unique_words
            self.index2word[self.n_unique_words] = word
            self.n_unique_words += 1
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_padding(self):
        # NOTE: should be called after finished with building vocab
        self.add_word(PAD_WORD)

    def populate(self, df):
        for rowid in tqdm(range(len(df))):
            df_row = df.iloc[rowid]
            for i in range(2):
                self.add_sentence(df_row.iloc[i])
        self.add_padding() # padding should be last in the vocabulary (for convenience in decoder)

class RecipeDataset(Dataset):
    def __init__(self, df, vocab):
        """
        Args:
            df (pd.DataFrame): dataframe with two columns: "Ingredients" and "Recipe"
            vocab (Vocabulary): to convert word2index
        """
        super().__init__()
        self.ingredient_recipe_df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.ingredient_recipe_df)
    
    def __getitem__(self, index):
        row = self.ingredient_recipe_df.iloc[index]
        ingredient_tens = torch.tensor([self.vocab.word2index[w] for w in row.Ingredients.split(" ")],
                                       dtype=torch.long, device=DEVICE)
        recipe_tens = torch.tensor([self.vocab.word2index[w] for w in row.Recipe.split(" ")],
                                       dtype=torch.long, device=DEVICE)
        return (ingredient_tens, recipe_tens)
    
# inspired by https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html
def pad_collate(vocab):

    def _pad_collate(batch):
        # print(len(batch))
        # print(batch[0])
        # ingredients: tuple of len batch_size with Tensor elements containing all ingredients in batch
        # recipes: tuple of len batch_size with Tensor elements containing all recipes in batch
        ingredients, recipes = zip(*batch)
        ingr_lens = torch.tensor([len(x) for x in ingredients], dtype=torch.long, device=DEVICE)
        recipe_lens = torch.tensor([len(r) for r in recipes], dtype=torch.long, device=DEVICE)

        ingredients_padded = pad_sequence(ingredients, batch_first=True, padding_value=vocab.word2index[PAD_WORD])
        recipes_padded = pad_sequence(recipes, batch_first=True, padding_value=vocab.word2index[PAD_WORD])

        return ingredients_padded, recipes_padded, ingr_lens, recipe_lens
    
    return _pad_collate

def pack(x_embed, x_lens):
    # convert tensor with padding to a PackedSequence, this allows rnns to ignore paddings
    return pack_padded_sequence(x_embed, x_lens.cpu().int(), batch_first=True, enforce_sorted=False)

def unpack(out_packed, padding_val):
    out_padded, out_lens = pad_packed_sequence(out_packed, batch_first=True, padding_value=padding_val)
    return out_padded, out_lens