import shutil
import os
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import argparse
import torch
import math
import sacrebleu
from tqdm import tqdm
from scipy.stats.mstats import gmean
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartConfig
)


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def preprocess_data(src_data, trg_data):
    df = pd.DataFrame({src_col: src_data[src_col], trg_col: trg_data[trg_col]})
    df = df.sample(frac=1)
    return df

def tokenize_datasets(df, src_col, trg_col, tokenizer, max_length):
    src_encodings = tokenizer(
        df[src_col].values.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    trg_encodings = tokenizer(
        df[trg_col].values.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length
    )
    dataset = CreateDataset(src_encodings, trg_encodings)
    return dataset

def train_model(model, train_dataset, dev_dataset, tokenizer, args):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.evaluate()

def generate_predictions(model, test_data, src_col, tokenizer, output_file):
    predictions = []
    for idx in range(len(test_data[src_col])):
        src = test_data[src_col].values[idx]
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=args.max_length)
        prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(prediction)

    with open(output_file, 'wb') as f:
        pickle.dump(remove_prefix(predictions, 'NEG: '), f)

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.labels['input_ids'])


def generate_synthetic_data(model, unparallel_data, src_col, tokenizer, synthetic_train_file):
    print(f"🔄 Generating synthetic train data from unparallel data...")
    predictions = []
    for idx in range(len(unparallel_data[src_col])):
        src = unparallel_data[src_col].values[idx]
        src_tknz = tokenizer(src, truncation=True, padding=True, max_length=args.max_length, return_tensors='pt')
        generated_ids = model.generate(src_tknz["input_ids"].cuda(), max_length=args.max_length)
        prediction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(prediction)

    data_pairs = pd.DataFrame({
            'pos': unparallel_data[src_col].values.tolist(),
            'neg': remove_prefix(predictions, 'NEG: ')
    })
    data_pairs.to_csv(synthetic_train_file, index=False)
    print(f"✅ Synthetic train data written to {synthetic_train_file} — {len(data_pairs)} pairs")


def compute_scores_for_synthetic_pairs(sources, preds, evaluator, target_label="NEGATIVE"):
    """
    Computes style probability, BLEU, and semantic similarity for each generated pair.
    Returns: list of dicts [{src,pred,style_prob,bleu,sim}]
    """
    print(f"🔄 Computing scores for synthetic pairs...")
    sent_pipe = evaluator.sentiment_analysis
    sim_model = evaluator.sim_model
    out = []
    for src, pred in tqdm(zip(sources, preds), total=len(preds), desc="Scoring pairs"):
        # 1️⃣ Style probability
        res = sent_pipe(pred)[0]
        prob = res['score']
        if res['label'] != target_label:
            style_prob = 1.0 - prob
        else:
            style_prob = prob

        # 2️⃣ BLEU (sentence-level vs source)
        bleu_vs_src = sacrebleu.sentence_bleu(pred, [src]).score / 100.0  # normalized 0–1

        # 3️⃣ Semantic similarity
        emb = sim_model.encode([src, pred])
        cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-9))
        sim_norm = (cos_sim + 1.0) / 2.0

        out.append({
            'src': src,
            'pred': pred,
            'style_prob': float(style_prob),
            'bleu': float(bleu_vs_src),
            'sim': float(sim_norm)
        })
    print(f"✅ Computer scores for the synthetic train data data.")
    return out


def compute_filtered_score(item, style_thresh=0.9):
    style_prob = item['style_prob']
    bleu = item['bleu']
    sim = item['sim']

    # 1️⃣ Filter out weak style transfers
    if style_prob < style_thresh:
        return 0.0
    
    # 2️⃣ Reward both semantic + structural similarity equally
    content_score = (bleu + sim) / 2.0
    
    # 3️⃣ Slightly emphasize very confident style transfers
    return float(content_score * (style_prob ** 1.5))


def add_scores_and_select(scored_list, top_k, high_thresh=5.5, low_thresh=5.0):
    """
    scored_list: list of dicts with keys ['style_prob', 'bleu', 'sim']
    top_k: number of examples to finally keep
    high_thresh: first (strict) threshold
    low_thresh: fallback threshold if too few examples above high_thresh

    Returns selected list sorted by score desc.
    """
    # Compute scores for all examples
    for it in scored_list:
        it['score'] = compute_filtered_score(it)

    # Sort descending by score
    scored_list.sort(key=lambda x: x['score'], reverse=True)

    # 1️⃣ Try selecting items above high threshold
    high_selected = [x for x in scored_list if x['score'] >= high_thresh]

    if len(high_selected) >= top_k:
        # We have enough high-quality examples
        return high_selected[:top_k]

    # 2️⃣ Not enough high-score examples → relax threshold
    low_selected = [x for x in scored_list if x['score'] >= low_thresh]

    # Return whichever is smaller (to avoid overshooting top_k)
    return low_selected[:min(len(low_selected), top_k)]



def write_evaluation_results(acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl, filepath):
    with open(filepath, 'w') as file:
        file.write('Accuracy: ' + str(acc) + '\n')
        file.write('Bleu with Source: ' + str(bleu_withsrc) + '\n')
        file.write('Bleu with Target: ' + str(bleu_withtrg) + '\n')
        file.write('Similarity: ' + str(sim) + '\n')
        file.write('GPT2 PPL: ' + str(gpt2_ppl) + '\n')

def remove_prefix(strings, prefix):
    return [string.replace(prefix, '') for string in strings]

def remove_newline(df):
    df = df.apply(lambda x: x.str.replace('\n', ''))
    return df

def main(args):
    shutil.rmtree('facebook', ignore_errors=True)
    set_seed(args.seed_value)

    train_df = remove_newline(pd.read_csv(args.train_file))
    train_df = train_df.sample(frac = 1, random_state = args.seed_value)

    dev_df = remove_newline(pd.read_csv(args.dev_file))
    dev_df = dev_df.sample(frac = 1, random_state = args.seed_value)

    test_df = remove_newline(pd.read_pickle(args.test_file))

    neg_prompt = 'NEG: '
    pos_prompt = 'POS: '
    if args.prompt_enabled:

        train_df["pos"] = train_df["pos"].apply(lambda x: pos_prompt+x)
        train_df["neg"] = train_df["neg"].apply(lambda x: neg_prompt+x)

        dev_df["pos"] = dev_df["pos"].apply(lambda x: pos_prompt+x)
        dev_df["neg"] = dev_df["neg"].apply(lambda x: neg_prompt+x)

        test_df["pos"] = test_df["pos"].apply(lambda x: pos_prompt+x)
        test_df["neg"] = test_df["neg"].apply(lambda x: neg_prompt+x)

    with open('../output/pos_to_neg/src.pkl', 'wb') as f:
        pickle.dump(remove_prefix(test_df[args.src_col].values.tolist(), pos_prompt), f)

    with open('../output/pos_to_neg/trg.pkl', 'wb') as f:
        pickle.dump(remove_prefix(test_df[args.trg_col].values.tolist(), neg_prompt), f)

    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    train_dataset = tokenize_datasets(train_df, args.src_col, args.trg_col, tokenizer, args.max_length)
    dev_dataset = tokenize_datasets(dev_df, args.src_col, args.trg_col, tokenizer, args.max_length)

    # Load the BART model
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    config = BartConfig.from_pretrained(args.model_name)
    config.dropout = args.dropout #0.15
    config.attention_dropout = args.attention_dropout #0.05
    config.activation_dropout = args.activation_dropout #0.05

    config.label_smoothing_factor = args.label_smoothing_factor #0.05

    model.config = config

    # Define a directory to save the model
    output_directory = "./results"

    # Initialize training arguments correctly
    training_args = Seq2SeqTrainingArguments(
      output_dir=output_directory,         # Explicitly name the output directory
      eval_strategy="epoch",         # This is correct for recent versions
      learning_rate=args.learning_rate,
      per_device_train_batch_size=args.batch_size,
      per_device_eval_batch_size=args.batch_size,
      weight_decay=args.weight_decay,
      save_total_limit=1,
      save_strategy='epoch',               # Note: save_strategy often aligns with evaluation_strategy
      load_best_model_at_end=True,
      num_train_epochs=args.num_train_epochs,
      predict_with_generate=True,
      fp16=True
    )

    # Train the model
    train_model(model, train_dataset, dev_dataset, tokenizer, training_args)

    # Generate predictions on test data
    generate_predictions(model, test_df, args.src_col, tokenizer, args.pred_file) ###
    print('Completed.')

    evaluator = AutomaticEvaluator(args.seed_value)
    evaluator.set_seed()
    evaluator.load_models()

    target_label = 'NEGATIVE' if args.task == 'pos_to_neg' else 'POSITIVE'

    # <-- CHANGED: Call evaluate_all with the new arguments from argparse
    acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl = evaluator.evaluate_all(
        srcs_file=args.src_file,
        refs_file=args.ref_file,
        preds_file=args.pred_file,
        target_label=target_label
    )

    # Print or use the evaluation results as needed
    write_evaluation_results(acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl, args.eval_scores_file)


if __name__ == "__main__":
    args = SimpleNamespace(
        seed_value=53,
        model_name="facebook/bart-base",
        max_length=128,
        batch_size=8,
        learning_rate=5e-5,
        weight_decay=0.0,
        num_train_epochs=3,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        label_smoothing_factor=0.0,

        train_file="../data/train.csv",
        dev_file="../data/dev.csv",
        test_file="../data/test.pkl",

        # --- CORRECTED SECTION ---
        # Renamed output_file to pred_file for clarity and added missing paths
        pred_file="../output/pos_to_neg/pred.pkl",
        src_file="../output/pos_to_neg/src.pkl",
        ref_file="../output/pos_to_neg/trg.pkl",
        # -------------------------

        eval_scores_file="../output/pos_to_neg/scores.txt",
        prompt_enabled=True,
        src_col="pos",
        trg_col="neg",
        dev_size=0.1,
        train_pkl_path="../output/pos_to_neg/train.pkl",
        task='pos_to_neg'
    )
    
    main(args)
