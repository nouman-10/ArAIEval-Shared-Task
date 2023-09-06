import logging
import logging.handlers
import argparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import jsonlines
import os

import sys
sys.path.append('.')
from format_checker.task1 import check_format, validate_files

"""
This script checks whether the results format for Task 1 (1A and 1B) is correct. 
It also provides some warnings about possible errors.

The submission of the result file for subtask 1A should be in TSV format. 
One row of the prediction file is the following:
id <TAB> label

where id is the tweet/paragraph id as given in the test file, and  label is the predicted binary label, either "true" or "false".
For example:

05001	false
05002	false
05003	true

=====================

The submission of the result file for subtask 1B should be in TSV format. 
One row of the prediction file is the following:
id <TAB> technique1,technique2,...,techniqueN

where id is the tweet/paragraph id as given in the test file, and technique1,technique2,...,techniqueN is a comma-separated list of techniques (or "no_technique"). 
For example:

05001	Loaded_Language
05002	no_technique
05003	Slogans,Straw_Man
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def read_techniques(file_path):
    with open(file_path) as f:
        techniques = [label.strip() for label in f.readlines()]

    return techniques


def _read_tsv_input_file(file_full_name):
    predictions = {}

    with open(file_full_name, encoding='utf-8') as f:
        logging.info("Skipping header row...")
        next(f) # Skip header row
        for line in f.readlines():
            cols = line.rstrip().split("\t")

            if len(cols) != 2:
                logging.error('The file must have two TAB seperated columns')
                return

            predictions[str(cols[0])] = cols[1].split(",")

    return predictions


def _read_gold_labels_file(gold_fpath, subtask):
    gold_labels = {}

    with jsonlines.open(gold_fpath) as gold_f:
        for obj in gold_f:
            if subtask == "1B":
                gold_labels[str(obj["id"])] = obj["labels"]
            else:
                gold_labels[str(obj["id"])] = obj["label"]

    return gold_labels


def _extract_matching_lists(pred_labels, gold_labels):
    """
  Extract the list of values from the two dictionaries ensuring that elements with the same key are in the same position.
  """
    pred_values, gold_values = ([], [])

    for k in gold_labels.keys():
        pred_values.append(pred_labels[k])
        gold_values.append(gold_labels[k])

    return pred_values, gold_values


def correct_labels(pred_labels, gold_labels):
    """
    Check if the labels in pred file match the gold labels.
    """
    if not len(pred_labels.keys()) == len(gold_labels.keys()):
        logging.error('Number of predictions (%d) is not the expected one (%d).' % (
            len(pred_labels.keys()), len(gold_labels.keys())))

        return False

    if not len(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys()))) == 0:

        logging.error('IDs of documents in prediction file don\'t match the gold labels file. Different IDs: ' + str(set(pred_labels.keys()).symmetric_difference(set(gold_labels.keys()))))

        return False

    return True


def evaluate(pred_labels, gold_labels, subtask, techniques=None):
    """
        Evaluates the predicted classes w.r.t. a gold file.
        Metrics are:  macro_f1 nd micro_f1
        :param pred_labels: a dictionary with predictions,
        :param gold_labels: a dictionary with gold labels.
     """
    pred_values, gold_values = _extract_matching_lists(pred_labels, gold_labels)

    # We are scoring for subtask 1B
    if subtask == "1B":
        mlb = MultiLabelBinarizer()
        mlb.fit([techniques])
        gold_values = mlb.transform(gold_values)
        pred_values = mlb.transform(pred_values)


    micro_f1 = f1_score(gold_values, pred_values, average="micro")
    macro_f1 = f1_score(gold_values, pred_values, average="macro")

    return micro_f1, macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file_path", "-g", required=True,
                        help="The absolute path to the gold labels file.", type=str)
    parser.add_argument("--pred_files_path", "-p", nargs='+', required=True,
                        help="The absolute path to the files of runs you want to score.", type=str)
    parser.add_argument("--subtask", "-s", required=True, choices=['1A', '1B', '2A', '2B'],
                        help="The subtask for which we are checking format: 1A, 1B, 2A, 2B", type=str)
    parser.add_argument("--techniques_file_path", "-c", required=False, default="./techniques_list_task1.txt",
                        help="In case of subtask 1B, the absolute path to the file containing all "
                             "possible persuasion techniques",
                        type=str)

    args = parser.parse_args()
    techniques_file_path = args.techniques_file_path
    pred_files_path = args.pred_files_path
    gold_file_path = args.gold_file_path
    subtask = args.subtask

    # Validate if files exist then score
    if validate_files(pred_files_path, subtask, techniques_file_path):
        techniques = []

        if not os.path.exists(gold_file_path):
            logging.error("File doesn't exist: {}".format(gold_file_path))
            exit()
        else:
            logging.info("All files exist!")


        gold_labels = _read_gold_labels_file(gold_file_path, subtask)

        if subtask.startswith("1B"):
            techniques = read_techniques(techniques_file_path)

        for pred_file_path in pred_files_path:
            logging.info("Checking file: {}".format(pred_file_path))

            # Check file format then score
            if check_format(pred_file_path, subtask, techniques):
                logging.info("Scoring run: {}".format(pred_file_path))

                pred_labels = _read_tsv_input_file(pred_file_path)

                if correct_labels(pred_labels, gold_labels):
                    micro_f1, macro_f1 = evaluate(pred_labels, gold_labels, subtask, techniques)

                    logging.info("micro-F1={:.4f}\tmacro-F1={:.4f}".format(micro_f1, macro_f1))
