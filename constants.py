import torch

ING_START = "<INGREDIENT_START>"
ING = "<INGREDIENT>"
ING_END = "<INGREDIENT_END>"
REC_START = "<RECIPE_START>"
REC = "<RECIPE_STEP>"
REC_END = "<RECIPE_END>"
SPECIAL_TAGS = {
    ING_START: 0,
    ING_END: 1,
    REC_START: 2,
    REC_END: 3,
    ING: 4,
    REC: 5,
}

PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNKNOWN>"
MAX_INGR_LEN = 150 # fixed from assignment
TEACHER_FORCING_RATIO = 1.0 # fixed from assignment

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")