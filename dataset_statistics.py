import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from eval import get_all_ingredients

# !pip install scikit-learn

# ! need to download this for tokenization
# nltk.download('punkt')
# nltk.download('stopwords')

def get_ds_statistics(df):
    all_ingredients_list = get_all_ingredients("./ingredient_set.json")

    df_ings_list = df.Ingredients.tolist()
    df_recipes_list = [r for r in df.Recipe.tolist() if not pd.isna(r)]

    print("Tokenizing...")

    tokenized_ings = [word_tokenize(i) for i in df_ings_list]
    tokenized_recipes = [word_tokenize(r) for r in df_recipes_list]

    ings_words = [i.split() for i in df_ings_list]
    recipes_words = [r.split() for r in df_recipes_list]

    ings_lens = [len(i) for i in ings_words]
    recipes_lens = [len(r) for r in recipes_words]

    print("Counting word frequencies...")
    stop_words = set(stopwords.words('english'))

    freqdist_all = [FreqDist() for _ in range(2)]
    freqdist_ing = FreqDist()

    for i, wordset in enumerate([tokenized_ings, tokenized_recipes]):
        for sample in wordset:
            for word in sample:
                if bool(re.match("\w+", word)) and word not in stop_words:
                    freqdist_all[i][word] += 1
                if i == 0 and word in all_ingredients_list:
                    freqdist_ing[word] += 1

    ## count number of ingredients per sample, i.e. how many ingredients are listed for each recipe
    num_ings_per_sample = []

    for i in tokenized_ings:
        num_ings_i = 0
        for word in i:
            num_ings_i += word in all_ingredients_list
        num_ings_per_sample.append(num_ings_i)

    def get_n_gram(n, l):
        cv = CountVectorizer(ngram_range=(n, n)).fit(l)
        bow = cv.transform(l)
        word_sum = bow.sum(axis=0)
        word_freqs = sorted([[word, word_sum[0, idx]] for word, idx in cv.vocabulary_.items()],
                            key=lambda w: w[1], reverse=True)
        return word_freqs
    
    print("Get N-grams...")
    ing_bigrams = get_n_gram(2, df_ings_list)
    ing_trigrams = get_n_gram(3, df_ings_list)

    rec_bigrams = get_n_gram(2, df_recipes_list)
    rec_trigrams = get_n_gram(3, df_recipes_list)

    print(f"Number of samples: {df.shape[0]}")
    print("===== INGREDIENTS =====\n"
          f"min. length: {min(ings_lens)}, max.length: {max(ings_lens)}, avg. length: {sum(ings_lens)/len(ings_lens):.2f}, "
          f"std. length: {np.std(ings_lens):.2f}\n"
          f"Max. number of ingredients: {max(num_ings_per_sample)}, min. number of ingredients: {min(num_ings_per_sample)}, "
          f"avg. number of ingredients: {sum(num_ings_per_sample)/len(num_ings_per_sample)}\n"
          f"10 most common tokens (excluding punuctuation and stopwords): {freqdist_all[0].most_common(10)}\n"
          f"10 most common ingredients: {freqdist_ing.most_common(10)}\n"
          f"Top 5 bigrams: {ing_bigrams[:5]}, top 5 trigrams: {ing_trigrams[:5]}")
    
    print("===== RECIPES =====\n"
          f"min. length: {min(recipes_lens)}, max.length: {max(recipes_lens)}, avg. length: {sum(recipes_lens)/len(recipes_lens):.2f}, "
          f"std. length: {np.std(recipes_lens):.2f}\n"
          f"10 most common tokens (excluding punuctuation and stopwords): {freqdist_all[1].most_common(10)}\n"
          f"Top 5 bigrams: {rec_bigrams[:5]}, top 5 trigrams: {rec_trigrams[:5]}")
    
    # print(f"(Recipes) min. length: {min(recipes_lens)}, max.length: {max(recipes_lens)}, avg. length: {sum(recipes_lens)/len(recipes_lens):.3f}, "
    #       f"std.length: {np.std(recipes_lens):.3f}")
    # print(f"Max. number of ingredients: {max(num_ings_per_sample)}, min. number of ingredients: {min(num_ings_per_sample)}, "
    #       f"avg. number of ingredients: {sum(num_ings_per_sample)/len(num_ings_per_sample)}")
    # print(f"10 most common tokens (excluding punctuation): {freqdist_all.most_common(10)}")
    # print(f"10 most common ingredients: {freqdist_ing.most_common(10)}")
    # print(f"Top 5 bigrams: {bigrams[:5]}, top 5 trigrams: {trigrams[:5]}")
