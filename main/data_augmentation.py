# !pip install nlpaug sentencepiece transformers datasets --quiet
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

import os, random, pandas as pd
import nlpaug.augmenter.word as naw
from tqdm import tqdm

random.seed(53)

# --- Load your data ---
train_path = "/content/text-style-transfer/data/train.csv"
dev_path   = "/content/text-style-transfer/data/dev.csv"

train_df = pd.read_csv(train_path)
dev_df   = pd.read_csv(dev_path)

# --- Create augmenters as used in the paper ---
augmenters = {
    "spelling": naw.SpellingAug(),
    "bert": naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute'),
    "synonym": naw.SynonymAug(aug_src='wordnet'),
    "swap": naw.RandomWordAug(action="swap"),
    "delete": naw.RandomWordAug(action="delete")
}

# --- Custom Split function (replacement for SplitAug) ---
def split_random_word(text, prob=0.1):
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 4 and random.random() < prob:
            cut = random.randint(1, len(w)-1)
            new_words.extend([w[:cut], w[cut:]])
        else:
            new_words.append(w)
    return " ".join(new_words)

# --- Apply random augmentation with 50% chance ---
def augment_df(df, prob=0.5):
    augmented_rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if random.random() < prob:
            aug_type = random.choice(list(augmenters.keys()) + ["split"])
            if aug_type == "split":
                new_pos = split_random_word(row["pos"])
                new_neg = split_random_word(row["neg"])
            else:
                aug = augmenters[aug_type]
                new_pos = aug.augment(row["pos"])
                new_neg = aug.augment(row["neg"])
            augmented_rows.append({"pos": new_pos, "neg": new_neg, "aug_type": aug_type})
    return pd.DataFrame(augmented_rows)

aug_train_df = augment_df(train_df)
aug_dev_df   = augment_df(dev_df)

print(f"Original train: {len(train_df)}, Augmented train: {len(aug_train_df)}")
print(f"Original dev: {len(dev_df)}, Augmented dev: {len(aug_dev_df)}")

# --- Combine and save ---
train_all = pd.concat([train_df, aug_train_df[["pos", "neg"]]], ignore_index=True)
dev_all   = pd.concat([dev_df, aug_dev_df[["pos", "neg"]]], ignore_index=True)

os.makedirs("/content/text-style-transfer/data_augmented", exist_ok=True)
train_all.to_csv("/content/text-style-transfer/data_augmented/train_aug.csv", index=False)
dev_all.to_csv("/content/text-style-transfer/data_augmented/dev_aug.csv", index=False)

print("✅ Augmented datasets saved in /content/text-style-transfer/data_augmented/")
