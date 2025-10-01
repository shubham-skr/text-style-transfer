import torch
import sys
import random
import numpy as np
import pickle
import argparse
import evaluate
import pandas as pd  # Added missing import
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, set_seed

class AutomaticEvaluator:
    def __init__(self, seed_value):
        self.seed_value = seed_value
        self.bleu = None
        self.sim_model = None
        self.sentiment_analysis = None
        self.gpt2_tokenizer = None
        self.gpt2_model = None

    def set_seed(self):
        random.seed(self.seed_value)
        np.random.seed(self.seed_value)
        torch.manual_seed(self.seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_value)
            torch.cuda.manual_seed_all(self.seed_value)
        set_seed(self.seed_value)

    def load_models(self):
        print("Loading evaluation models...")
        self.bleu = evaluate.load("bleu")
        self.sim_model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.sentiment_analysis = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

        with torch.no_grad():
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_model.eval()
        print("Models loaded successfully.")

    def gpt2_score(self, sentence):
        tokenize_input = self.gpt2_tokenizer.encode(sentence, return_tensors='pt')
        with torch.no_grad():
            loss = self.gpt2_model(tokenize_input, labels=tokenize_input).loss
        return torch.exp(loss).item()

    def similarity(self, sentence1, sentence2):
        # Sentences can be encoded in batches for efficiency
        embeddings = self.sim_model.encode([sentence1, sentence2])
        sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])
        return sim_score[0][0]

    def senti_score(self, sentence):
        return self.sentiment_analysis(sentence)[0]

    # <-- CHANGED: Method signature now accepts file paths
    def evaluate_all(self, srcs_file, refs_file, preds_file, target_label):
        # <-- CHANGED: Load data from pickle files
        with open(srcs_file, 'rb') as f:
            sources = pickle.load(f)
        with open(refs_file, 'rb') as f:
            references = pickle.load(f)
        with open(preds_file, 'rb') as f:
            predictions = pickle.load(f)

        print(f"Evaluating {len(predictions)} predictions...")

        # Prepare data for BLEU score calculation
        references_bleu = [[ref] for ref in references]
        sources_bleu = [[src] for src in sources]

        sim_scores = []
        gpt2_scores = []
        correct_count = 0

        for i in range(len(predictions)):
            pred = predictions[i]
            ref = references[i]

            # 1. Accuracy Score
            if self.senti_score(pred)['label'] == target_label:
                correct_count += 1

            # 2. Similarity Score
            sim_scores.append(self.similarity(pred, ref))

            # 3. Fluency Score (Perplexity)
            gpt2_scores.append(self.gpt2_score(pred))

        # 4. BLEU Scores (calculated in batch)
        bleu_withsrc = self.bleu.compute(predictions=predictions, references=sources_bleu, max_order=4)
        bleu_withtrg = self.bleu.compute(predictions=predictions, references=references_bleu, max_order=4)

        # Calculate final averages
        acc = correct_count / len(predictions)
        avg_sim = sum(sim_scores) / len(sim_scores)
        avg_gpt2_ppl = sum(gpt2_scores) / len(gpt2_scores)

        return (
            acc,
            bleu_withsrc['bleu'],
            bleu_withtrg['bleu'],
            avg_sim,
            avg_gpt2_ppl
        )


def main(args):
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

    print('-' * 20)
    print(f"Evaluation Results for Task: {args.task}")
    print('-' * 20)
    print(f"Accuracy: {acc:.4f}")
    print(f"Bleu with Source: {bleu_withsrc:.4f}")
    print(f"Bleu with Target: {bleu_withtrg:.4f}")
    print(f"Similarity: {sim:.4f}")
    print(f"GPT-2 PPL (Fluency): {gpt2_ppl:.4f}")



# import torch
# import sys
# import random
# import numpy as np
# import pickle
# import argparse
# import evaluate
# import pandas as pd  # Added missing import
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, set_seed

# class AutomaticEvaluator:
#     def __init__(self, seed_value):
#         self.seed_value = seed_value
#         self.bleu = None
#         self.sim_model = None
#         self.sentiment_analysis = None
#         self.gpt2_tokenizer = None
#         self.gpt2_model = None

#     def set_seed(self):
#         random.seed(self.seed_value)
#         np.random.seed(self.seed_value)
#         torch.manual_seed(self.seed_value)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(self.seed_value)
#             torch.cuda.manual_seed_all(self.seed_value)
#         set_seed(self.seed_value)

#     def load_models(self):
#         print("Loading evaluation models...")
#         self.bleu = evaluate.load("bleu")
#         self.sim_model = SentenceTransformer('bert-base-nli-mean-tokens')
#         self.sentiment_analysis = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

#         with torch.no_grad():
#             self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#             self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
#             self.gpt2_model.eval()
#         print("Models loaded successfully.")

#     def gpt2_score(self, sentence):
#         tokenize_input = self.gpt2_tokenizer.encode(sentence, return_tensors='pt')
#         with torch.no_grad():
#             loss = self.gpt2_model(tokenize_input, labels=tokenize_input).loss
#         return torch.exp(loss).item()

#     def similarity(self, sentence1, sentence2):
#         # Sentences can be encoded in batches for efficiency
#         embeddings = self.sim_model.encode([sentence1, sentence2])
#         sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])
#         return sim_score[0][0]

#     def senti_score(self, sentence):
#         return self.sentiment_analysis(sentence)[0]

#     # <-- CHANGED: Method signature now accepts file paths
#     def evaluate_all(self, srcs_file, refs_file, preds_file, target_label):
#         # <-- CHANGED: Load data from pickle files
#         with open(srcs_file, 'rb') as f:
#             sources = pickle.load(f)
#         with open(refs_file, 'rb') as f:
#             references = pickle.load(f)
#         with open(preds_file, 'rb') as f:
#             predictions = pickle.load(f)

#         print(f"Evaluating {len(predictions)} predictions...")

#         # Prepare data for BLEU score calculation
#         references_bleu = [[ref] for ref in references]
#         sources_bleu = [[src] for src in sources]
        
#         sim_scores = []
#         gpt2_scores = []
#         correct_count = 0

#         for i in range(len(predictions)):
#             pred = predictions[i]
#             ref = references[i]
            
#             # 1. Accuracy Score
#             if self.senti_score(pred)['label'] == target_label:
#                 correct_count += 1
            
#             # 2. Similarity Score
#             sim_scores.append(self.similarity(pred, ref))
            
#             # 3. Fluency Score (Perplexity)
#             gpt2_scores.append(self.gpt2_score(pred))

#         # 4. BLEU Scores (calculated in batch)
#         bleu_withsrc = self.bleu.compute(predictions=predictions, references=sources_bleu, max_order=4)
#         bleu_withtrg = self.bleu.compute(predictions=predictions, references=references_bleu, max_order=4)

#         # Calculate final averages
#         acc = correct_count / len(predictions)
#         avg_sim = sum(sim_scores) / len(sim_scores)
#         avg_gpt2_ppl = sum(gpt2_scores) / len(gpt2_scores)

#         return (
#             acc,
#             bleu_withsrc['bleu'],
#             bleu_withtrg['bleu'],
#             avg_sim,
#             avg_gpt2_ppl
#         )


# def main(args):
#     evaluator = AutomaticEvaluator(args.seed_value)
#     evaluator.set_seed()
#     evaluator.load_models()

#     target_label = 'NEGATIVE' if args.task == 'pos_to_neg' else 'POSITIVE'

#     # <-- CHANGED: Call evaluate_all with the new arguments from argparse
#     acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl = evaluator.evaluate_all(
#         srcs_file=args.src_file,
#         refs_file=args.ref_file,
#         preds_file=args.pred_file,
#         target_label=target_label
#     )

#     print('-' * 20)
#     print(f"Evaluation Results for Task: {args.task}")
#     print('-' * 20)
#     print(f"Accuracy: {acc:.4f}")
#     print(f"Bleu with Source: {bleu_withsrc:.4f}")
#     print(f"Bleu with Target: {bleu_withtrg:.4f}")
#     print(f"Similarity: {sim:.4f}")
#     print(f"GPT-2 PPL (Fluency): {gpt2_ppl:.4f}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate text style transfer models.")

#     parser.add_argument('--seed_value', type=int, default=333)
#     parser.add_argument('--task', type=str, choices=['pos_to_neg', 'neg_to_pos'], required=True)
    
#     # <-- CHANGED: argparse now takes explicit file paths
#     parser.add_argument('--src_file', type=str, required=True, help="Path to the source sentences .pkl file")
#     parser.add_argument('--ref_file', type=str, required=True, help="Path to the reference sentences .pkl file")
#     parser.add_argument('--pred_file', type=str, required=True, help="Path to the predicted sentences .pkl file")
    
#     args = parser.parse_args()
#     main(args)


# # import torch
# # import sys
# # import random
# # import numpy as np
# # import pickle
# # import argparse
# # import evaluate
# # from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity
# # from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForMaskedLM, set_seed

# # class AutomaticEvaluator:
# #     def __init__(self, seed_value):
# #         self.seed_value = seed_value
# #         self.bleu = None
# #         self.sim_model = None
# #         self.sentiment_analysis = None
# #         self.gpt2_tokenizer = None
# #         self.gpt2_model = None

# #     def set_seed(self):
# #         random.seed(self.seed_value)
# #         np.random.seed(self.seed_value)
# #         torch.manual_seed(self.seed_value)
# #         torch.cuda.manual_seed(self.seed_value)
# #         torch.cuda.manual_seed_all(self.seed_value)
# #         set_seed(self.seed_value)

# #     def load_models(self):
# #         self.bleu = evaluate.load("bleu")
# #         self.sim_model = SentenceTransformer('bert-base-nli-mean-tokens')
# #         self.sim_model.random_state = self.seed_value
# #         self.sentiment_analysis = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')
# #         self.sentiment_analysis.random_state = self.seed_value

# #         with torch.no_grad():
# #             self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# #             self.gpt2_tokenizer.random_state = self.seed_value
# #             self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
# #             self.gpt2_model.random_state = self.seed_value
# #             self.gpt2_model.eval()

# #     def gpt2_score(self, sentence):
# #         tokenize_input = self.gpt2_tokenizer.encode(sentence)
# #         tensor_input = torch.tensor([tokenize_input])
# #         loss = self.gpt2_model(tensor_input, labels=tensor_input)[0]
# #         return np.exp(loss.detach().numpy())

# #     def similarity(self, sentence1, sentence2):
# #         sentence_embedding1 = self.sim_model.encode(sentence1)
# #         sentence_embedding2 = self.sim_model.encode(sentence2)
# #         sim_score = cosine_similarity([sentence_embedding1], [sentence_embedding2])
# #         return sim_score[0][0]

# #     def senti_score(self, sentence):
# #         return self.sentiment_analysis(sentence)[0]

# #     def evaluate_all(self, output, target_label):
# #         sources = list()
# #         references = list()
# #         predictions = list()

# #         with open(srcs, 'rb') as f:
# #             sources = output['src'].to_list()
# #         with open(refs, 'rb') as f:
# #             references = output['trg'].to_list()
# #         with open(preds, 'rb') as f:
# #             predictions = output['pred'].to_list()

# #         print(len(predictions))

# #         sources_bleu = list()
# #         references_bleu = list()
# #         sim_scores = list()
# #         gpt2_scores = list()
# #         correct_count = 0

# #         for idx, reference in enumerate(references):
# #             references_bleu.append([reference])
# #             sources_bleu.append([sources[idx]])

# #             if self.senti_score(predictions[idx])['label'] == target_label:
# #                 correct_count += 1

# #             sim_scores.append(self.similarity(predictions[idx], reference))
# #             gpt2_scores.append(self.gpt2_score(predictions[idx]))

# #         bleu_withsrc = self.bleu.compute(predictions=predictions, references=sources_bleu, max_order=4)
# #         bleu_withtrg = self.bleu.compute(predictions=predictions, references=references_bleu, max_order=4)

# #         return (
# #             correct_count / len(predictions),
# #             bleu_withsrc['bleu'],
# #             bleu_withtrg['bleu'],
# #             sum(sim_scores) / len(sim_scores),
# #             sum(gpt2_scores) / len(gpt2_scores)
# #         )


# # def main(args):
# #     evaluator = AutomaticEvaluator(args.seed_value)
# #     evaluator.set_seed()
# #     evaluator.load_models()

# #     target_label = None

# #     if args.task == 'pos_to_neg':
# #         target_label = 'NEGATIVE'
# #     elif args.task == 'neg_to_pos':
# #         target_label = 'POSITIVE'

# #     output = pd.read_csv(args.output_file)
# #     acc, bleu_withsrc, bleu_withtrg, sim, gpt2_ppl = evaluator.evaluate_all(output, target_label)

# #     print(args.task)
# #     print('-' * 10)
# #     print('Accuracy', acc)
# #     print('Bleu with Source', bleu_withsrc)
# #     print('Bleu with Target', bleu_withtrg)
# #     print('Similarity', sim)
# #     print('GPT2 PPL', gpt2_ppl)


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()

# #     parser.add_argument('--seed_value', type=int, default=333)
# #     parser.add_argument('--task', type=str, choices=['pos_to_neg', 'neg_to_pos'], default='pos_to_neg')
# #     parser.add_argument('--output_file', type=str, required=True)
# #     args = parser.parse_args()
# #     main(args)
