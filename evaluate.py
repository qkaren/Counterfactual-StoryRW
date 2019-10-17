import argparse
import json
from typing import List

import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
import spacy
import tqdm
import numpy as np
import rouge
import edlib
import os
import pandas as pd
import re
import glob

from pytorch_pretrained_bert import BertTokenizer, BertModel
from wmd import WMD
from torch.nn.modules.distance import CosineSimilarity

torch_emb_sim = CosineSimilarity()

from bert_score import score as bert_score


nlp = spacy.load("en_core_web_md")
nlp.add_pipe(WMD.SpacySimilarityHook(nlp), last=True)


def _clean_text(txt):
    return txt.lower()


class CFRInstance(object):
    def __init__(self,
                 original_context: str,
                 cf_context: str,
                 original_ending: str,
                 predicted_ending: str,
                 gold_cf_endings: List[str],
                 ):
        self.original_context = original_context
        self.cf_context = cf_context

        self.predicted_ending = predicted_ending
        self.original_ending = original_ending
        self.gold_cf_endings = gold_cf_endings

        self.spacy_docs = {
            'original_context': nlp(_clean_text(self.original_context)),
            'original_ending': nlp(_clean_text(self.original_ending)),
            'cf_context': nlp(_clean_text(self.cf_context)),
            'predicted_ending': nlp(_clean_text(self.predicted_ending)),
            'gold_cf_endings': [nlp(_clean_text(g)) for g in self.gold_cf_endings]
        }

        self.original_context_tokens = [t.text for t in self.spacy_docs['original_context']]
        self.original_ending_tokens = [t.text for t in self.spacy_docs['original_ending']]
        self.cf_context_tokens = [t.text for t in self.spacy_docs['cf_context']]
        self.predicted_ending_tokens = [t.text for t in self.spacy_docs['predicted_ending']]
        self.gold_cf_endings_tokens = [[t.text for t in _spacy_doc] for _spacy_doc in
                                       self.spacy_docs['gold_cf_endings']]


def read_lines(filename):
    lines = []
    with open(filename, "r") as f:
        for line in tqdm.tqdm(f):
            l = line.strip()
            if len(re.sub(r'[^\w\s]', '', l)) == 0:
                lines.append("")
            else:
                lines.append(l)
    return lines


def read_jsonl_lines(filename):
    with open(filename) as f:
        for line in tqdm.tqdm(f):
            yield json.loads(line.strip())


def _read_gold_cf_endings(gold_cf_endings_dir):
    gold_cf_endings = []
    for f in os.listdir(gold_cf_endings_dir):
        df = pd.read_csv(os.path.join(gold_cf_endings_dir, f))
        new_3 = df["new_3"]
        new_4 = df["new_4"]
        new_5 = df["new_5"]

        gold_cf_ending_lst = []

        for s3, s4, s5 in zip(new_3, new_4, new_5):
            gold_cf_ending_lst.append(' '.join([s3, s4, s5]))
        gold_cf_endings.append(gold_cf_ending_lst)

    return [list(Z) for Z in zip(*gold_cf_endings)]


def eval_bleu(instances: List[CFRInstance]):
    references = []
    hypotheses = []
    for instance in tqdm.tqdm(instances):
        references.append(instance.gold_cf_endings_tokens)
        hypotheses.append(instance.predicted_ending_tokens)

    corpus_bleu_scores = corpus_bleu(
        references, hypotheses, smoothing_function=SmoothingFunction().method4
    )

    sentence_bleu_scores = []
    total_skipped = 0
    for r, h in tqdm.tqdm(zip(references, hypotheses)):
        if len(h) == 0:
            sentence_bleu_scores.append(0)
            continue
        else:
            try:
                sentence_bleu_scores.append(
                    sentence_bleu(r, h, smoothing_function=SmoothingFunction().method4))
            except:
                sentence_bleu_scores.append(0.0)
                total_skipped+=1

    print("Total skipped = {}".format(total_skipped))

    metrics = {
        'corpus_bleu': corpus_bleu_scores,
        'mean_sentence_bleu': np.mean(sentence_bleu_scores),
        'sentence_bleu_by_instance': sentence_bleu_scores
    }
    return metrics


def eval_rouge(instances: List[CFRInstance]):
    references = []
    hypotheses = []

    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=4,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=True,
                            apply_best=False,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    by_instance = []
    for instance in instances:
        _r = [_clean_text(g) for g in instance.gold_cf_endings]
        _h = _clean_text(instance.predicted_ending)
        references.append(_r)
        hypotheses.append(_h)
        try:
            by_instance.append(evaluator.get_scores(_h, _r))
        except:
            by_instance.append({})

    scores = evaluator.get_scores(hypotheses, references)
    return {'rouge_all' : scores,
            'rouge_by_instance': by_instance
            }


def eval_bert_score(instances: List[CFRInstance], bert_model="bert-base-uncased"):
    references = []
    hypotheses = []
    for instance in instances:
        clean_reference = _clean_text(instance.original_context + ' ' + instance.original_ending)
        clean_hypothesis = _clean_text(instance.cf_context + ' ' + instance.predicted_ending)
        if len(clean_hypothesis) == 0:
            continue
        references.append(clean_reference)
        hypotheses.append(clean_hypothesis)

    P, R, F1 = bert_score(hypotheses, references, bert=bert_model, verbose=True)
    return {
        "bert_score_P": P.mean().item(),
        "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item(),
        "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def cigar_to_word_sets(cigar_path: str, modification_lst, reference_lst):
    diff_counts = [int(c) for c in re.split("[D=IX]", cigar_path) if c != '']
    diff_types = [c for c in cigar_path if c in {'D', '=', 'I', 'X'}]

    reference_ctr = 0
    modification_ctr = 0

    deleted_items = []
    inserted_items = []
    replaced_items = []
    equal_items = []

    deleted_counts = 0
    inserted_counts = 0
    replaced_counts = 0
    equal_counts = 0

    for count, type in zip(diff_counts, diff_types):
        if type == "D":
            # Deleted token
            for i in range(count):
                deleted_items.append(reference_lst[reference_ctr])
                reference_ctr += 1
            deleted_counts += count
        elif type == "I":
            # Inserted token
            for i in range(count):
                inserted_items.append(modification_lst[modification_ctr])
                modification_ctr += 1
            inserted_counts += count
        elif type == "=":
            # Same token
            for i in range(count):
                equal_items.append(reference_lst[reference_ctr])
                reference_ctr += 1
                modification_ctr += 1
            equal_counts += count
        elif type == "X":
            # Exchanged items
            for i in range(count):
                replaced_items.append(
                    (reference_lst[reference_ctr], modification_lst[modification_ctr]))
                reference_ctr += 1
                modification_ctr += 1
            replaced_counts += count

    return {
        "deleted": deleted_items,
        "inserted": inserted_items,
        "replaced": replaced_items,
        "equal": equal_items
    }


def compare_edit_sets_unigram(predicted_edits, reference_edits):
    keys = ["deleted", "inserted", "replaced", "equal"]

    total_edits = np.sum([len(set(reference_edits[k])) for k in keys])

    equivalent_edits_count = 0
    for k in keys:
        equivalent_edits_count += len(
            set(reference_edits[k]).intersection(set(predicted_edits[k])))

    return equivalent_edits_count / total_edits


def compare_edit_sets(predicted_edits, reference_edits, method="unigram"):
    if method == "unigram":
        return compare_edit_sets_unigram(predicted_edits, reference_edits)


def eval_rewrite(instances: List[CFRInstance]):
    instance_score = []

    for instance in tqdm.tqdm(instances):
        if len(instance.predicted_ending_tokens) == 0:
            instance_score.append(0)
            continue
        predicted_edits = edlib.align(
            instance.predicted_ending_tokens,
            instance.original_ending_tokens,
            mode="NW",
            task="path"
        )
        predicted_edits_set = \
            cigar_to_word_sets(predicted_edits['cigar'],
                               instance.predicted_ending_tokens,
                               instance.original_ending_tokens
                               )

        scores = []
        gold_edit_sets = []
        for gold_cf in instance.gold_cf_endings_tokens:
            gold_edits = edlib.align(
                gold_cf,
                instance.original_ending_tokens,
                mode="NW",
                task="path"
            )
            gold_edits_set = \
                cigar_to_word_sets(gold_edits['cigar'],
                                   gold_cf,
                                   instance.original_ending_tokens
                                   )
            gold_edit_sets.append(gold_edits_set)

            scores.append(compare_edit_sets(predicted_edits_set, gold_edits_set))

        instance_score.append(max(scores))

    return {
        'CFR_METRIC': np.mean(instance_score),
        'CFR_METRIC_by_instance': instance_score
    }


def eval_wmd(instances: List[CFRInstance]):
    wmd_scores = []
    for instance in instances:
        pred_ending_spacy_doc = instance.spacy_docs['predicted_ending']
        wmd_scores.append(
            np.min(
                [pred_ending_spacy_doc.similarity(gold_spacy_doc)
                 for gold_spacy_doc in instance.spacy_docs['gold_cf_endings']]
            )
        )
    return {
        'mean_wmd': np.mean(wmd_scores),
        'wmd_by_instance': wmd_scores
    }


def _bert_embed_sentence(sentence, bert_model: BertModel, bert_tokenizer: BertTokenizer):
    text = "[CLS] {} [SEP]".format(sentence)
    tokenized_text = bert_tokenizer.tokenize(text)
    indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids = [0] * len(indexed_tokens)
    segments_tensors = torch.tensor([segments_ids])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokens_tensor = tokens_tensor.to(device)
    segments_tensors = segments_tensors.to(device)

    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors, output_all_encoded_layers=False)

    # Embedding of the [CLS] token
    return encoded_layers[0][0]


def drift_similarity(original_story_emb, predicted_ending_emb, gold_cf_emb):
    drift_1 = predicted_ending_emb - original_story_emb
    drift_2 = gold_cf_emb - original_story_emb
    return torch_emb_sim(drift_1.unsqueeze(0), drift_2.unsqueeze(0)).item()


def eval_semantic_sim_score(instances: List[CFRInstance], bert_model_type="bert-base-uncased"):

    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    model = BertModel.from_pretrained(bert_model_type)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    drift_similarities = []

    for instance in instances:
        clean_original_story = _clean_text(instance.original_context + ' ' + instance.original_ending)
        predicted_ending = _clean_text(instance.cf_context + ' ' + instance.predicted_ending)

        original_story_emb = _bert_embed_sentence(clean_original_story, model, tokenizer)
        predicted_ending_emb = _bert_embed_sentence(predicted_ending, model, tokenizer)

        all_sims = []
        for gold_cf in instance.gold_cf_endings:
            clean_gold_cf = _clean_text(instance.cf_context + ' ' + gold_cf)
            gold_cf_emb = _bert_embed_sentence(clean_gold_cf, model, tokenizer)

            all_sims.append(drift_similarity(original_story_emb, predicted_ending_emb, gold_cf_emb))

        drift_similarities.append(np.max(all_sims))

    return {
        "drift_similarity": np.mean(drift_similarities),
        "drift_similarity_by_instance": [float(f) for f in  drift_similarities]
    }


def main(pred_endings_file, gold_file, bert_model):
    pred_endings = read_lines(filename=pred_endings_file)
    gold_records = read_jsonl_lines(gold_file)

    instances = []
    for pe, record in tqdm.tqdm(
            zip(pred_endings, gold_records)
    ):
        instance = CFRInstance(
            original_context=record['ori_context'],
            cf_context=record['cf_context'],
            predicted_ending=pe,
            original_ending=' '.join(record['ori_endinng']),
            gold_cf_endings=[' '.join(_ge) for _ge in record['gold_end']]
        )
        instances.append(instance)
    metrics = {}
    print("Eval BLEU ... ")
    metrics.update(eval_bleu(instances))
    print("Eval ROUGE ... ")
    metrics.update(eval_rouge(instances))
    print("Eval BertScore ... ")
    metrics.update(eval_bert_score(instances, bert_model=bert_model))
    # print("Eval CFRScore ... ")
    # metrics.update(eval_rewrite(instances))
    # print("Eval WMD ... ")
    # metrics.update(eval_wmd(instances))
    print("Eval Drift Similarity ... ")
    metrics.update(eval_semantic_sim_score(instances, bert_model_type=bert_model))
    print(metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluate.py',
        usage='%(prog)s gold_annotations predictions',
        description='Evaluate story rewrite'
    )

    parser.add_argument('--pred-endings-file', type=str,
                        dest="pred_endings_file",
                        help='Location of prediction file. Usually named *_pred.txt',
                        default=None)

    parser.add_argument('--all-preds-dir', type=str,
                        dest="all_preds_dir",
                        help='Location of prediction file. Usually named *_pred.txt',
                        default=None)

    parser.add_argument('--gold-file', type=str,
                        dest="gold_file",
                        help='Location of human annotated cf endings and rest of the data. Usually named [dev/test].jsonl')

    parser.add_argument('--bert_model', type=str,
                        dest="bert_model",
                        help='Location of human annotated cf endings and rest of the data. Usually named [dev/test].jsonl',
                        default=None)

    parser.add_argument('--output_file', type=str,
                        dest="output_file",
                        help='')

    args = parser.parse_args()

    # Run seed selection if args valid
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")

    assert args.all_preds_dir is not None or args.pred_endings_file is not None

    all_metrics = {}
    if args.all_preds_dir is not None:
        for f in glob.iglob(args.all_preds_dir + "/*/*.txt"):
            print("Processing file {}".format(f))
            metrics = main(f, args.gold_file, args.bert_model)
            model_name = os.path.basename(f).split(".")[0]
            all_metrics[model_name] = metrics

    else:
        all_metrics = main(args.pred_endings_file, args.gold_file, args.bert_model)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(all_metrics))
        f.close()
