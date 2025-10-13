import os, random, pickle
import pandas as pd

random.seed(53)

ref0_path = "/content/Sentiment-and-Style-Transfer/data/yelp/reference.0"  # neg\tpos
ref1_path = "/content/Sentiment-and-Style-Transfer/data/yelp/reference.1"  # pos\tneg

def read_tsv_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    return lines

lines0 = read_tsv_lines(ref0_path)
lines1 = read_tsv_lines(ref1_path)

pairs = []
for ln in lines1:
    if '\t' not in ln:
        continue
    left, right = ln.split('\t', 1)
    pos = left.strip()
    neg = right.strip()
    pairs.append({'pos': pos, 'neg': neg})

for ln in lines0:
    if '\t' not in ln:
        continue
    left, right = ln.split('\t', 1)
    neg = left.strip()
    pos = right.strip()
    pairs.append({'pos': pos, 'neg': neg})

print("Total pairs combined:", len(pairs)) 

random.shuffle(pairs)

# Paper split: 400 train, 100 dev, 500 test
train_pairs = pairs[:400]
dev_pairs   = pairs[400:500]
test_pairs  = pairs[500:1000]

train_df = pd.DataFrame(train_pairs)
dev_df   = pd.DataFrame(dev_pairs)
test_df  = pd.DataFrame(test_pairs)

os.makedirs("/content/text-style-transfer/data", exist_ok=True)
train_df.to_csv("/content/text-style-transfer/data/train.csv", index=False)
dev_df.to_csv("/content/text-style-transfer/data/dev.csv", index=False)
test_df.to_pickle("/content/text-style-transfer/data/test.pkl")   

print("Wrote data/train.csv, data/dev.csv, data/test.pkl")
print("Train size:", len(train_df), "Dev size:", len(dev_df), "Test size:", len(test_df))


# Unparallel Data
unparallel_ref1_path = "/content/Sentiment-and-Style-Transfer/data/yelp/sentiment.dev.1"  #
lines = []
with open(unparallel_ref1_path, 'r', encoding='utf-8') as f:
    lines = [ln.strip() for ln in f if ln.strip()]
print("Total unparallel lines:", len(lines))

unparallel_data_df = pd.DataFrame({'pos': lines})
unparallel_data_df.to_pickle("/content/text-style-transfer/data/unparallel.pkl")
print("Wrote data/unparallel.pkl")
