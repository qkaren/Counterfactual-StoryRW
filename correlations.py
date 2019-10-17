import argparse
import json
import pandas as pd
from scipy.stats import pearsonr

def main(human_ann_file, metrics_file):

    metrics = json.load(open(metrics_file))
    metrics['gold_1'] = metrics['g1']
    metrics['gold_2'] = metrics['g2']
    metrics['gold_3'] = metrics['g3']
    human_ann_df = pd.read_csv(human_ann_file)
    unique_models = list(human_ann_df["Input.MODEL"].unique())

    eval_categories = [
        'Answer.counterfactual',
        'Answer.ending',
        'Answer.plot',
        'Answer.premise',
        'Answer.second'
    ]

    print("{}\t{}\t{}\t{}".format("Model Name", "Human Acc", "Drift Similarity", "CFR"))
    for ec in eval_categories:
        print("`======= {} ".format(ec))

        human_eval_numbers = []
        drift_similarities = []
        cfr_metrics = []

        for um in unique_models:
            if um == "gold_1" or um == "gold_2":
                continue
            model_df = human_ann_df[human_ann_df["Input.MODEL"] == um]
            model_story_ann = model_df.groupby("Input.STORY").aggregate("mean")
            ec_accuracy = (model_story_ann[ec] >= 2).mean()
            original_name = um.split(".")[0]
            if original_name not in metrics:
                print("SKIPPING {}".format(original_name))
                continue
            print("{}\t{}\t{}\t{}".format(um, ec_accuracy, metrics[original_name]["drift_similarity"], metrics[original_name]["CFR_METRIC"]))
            human_eval_numbers.append(ec_accuracy)
            drift_similarities.append(metrics[original_name]["drift_similarity"])
            cfr_metrics.append(metrics[original_name]["CFR_METRIC"])

        drift_correl = pearsonr(human_eval_numbers, drift_similarities)
        cfr_correl = pearsonr(human_eval_numbers, cfr_metrics)
        print("DRIFT:\tCorrelation:\t{}\tP-value\t{}".format(drift_correl[0], drift_correl[1]))
        print("CFR:\tCorrelation:\t{}\tP-value\t{}".format(cfr_correl[0], cfr_correl[1]))
        print("\n\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='evaluate.py',
        usage='%(prog)s gold_annotations predictions',
        description='Evaluate story rewrite'
    )

    parser.add_argument('--human-ann-file', type=str,
                        dest="human_ann_file",
                        help='Location of human annotation file. Usually obtained from mturk download and named *.csv',
                        default=None)

    parser.add_argument('--metrics-file', type=str,
                        dest="metrics_file",
                        help='Location of metrics file. Usually named metrics.json',
                        default=None)

    args = parser.parse_args()

    # Run seed selection if args valid
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")

    main(args.human_ann_file, args.metrics_file)
