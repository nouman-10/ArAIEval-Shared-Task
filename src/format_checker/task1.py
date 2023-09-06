import os
import argparse
import logging
from pathlib import Path

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


def check_format(file_path, subtask, techniques=None):
    logging.info("Checking format of prediction file...")

    if not os.path.exists(file_path):
        logging.error("File doesn't exist: {}".format(file_path))
        return False

    with open(file_path, encoding='UTF-8') as out:
        next(out)
        file_content = out.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            doc_id, labels = line.strip().split('\t')

            if subtask == "1A":
                if labels not in ["true", "false"]:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False

            if subtask == "2A":
                if labels not in ["disinfo", "no-disinfo"]:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False


            if subtask == "1B":
                for label in labels.split(","):
                    label = label.strip()
                    if label not in techniques:
                        logging.error("Unknown label {} in line {}".format(label, i))
                        return False

            if subtask == "2B":
                if labels not in ["OFF", "Rumor", "SPAM", "HS"]:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False


    logging.info("File passed format checker!")

    return True


def validate_files(pred_files_path, subtask, techniques_file_path):
    logging.info("Validating if passed files exist...")

    if subtask.startswith("1B"):
        if not (techniques_file_path):
            logging.error("The path to file listing all persuasion techniques must be provided")
            return False

        if not os.path.exists(techniques_file_path):
            logging.error("File doesn't exist: {}".format(techniques_file_path))
            return False

    for pred_file_path in pred_files_path:
        if not os.path.exists(pred_file_path):
            logging.error("File doesn't exist: {}".format(pred_file_path))
            return False

        # Check if the filename matches what is required by the task
        subtasks = ["task1A", "task1B", 'task2A', 'task2B']
        if not any(Path(pred_file_path).name.startswith(st_name) for st_name in subtasks):
            logging.error("The submission file must start by task name! possible prefixes: " + str(subtasks))
            return False


    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files_path", "-p", nargs='+', required=True,
                        help="The absolute path to the files you want to check.", type=str)
    parser.add_argument("--subtask", "-s", required=True, choices=['1A', '1B', '2A', '2B'],
                        help="The subtask for which we are checking format: 1A, 1B, 2A, 2B", type=str)
    parser.add_argument("--techniques_file_path", "-c", required=False,
                        help="In case of subtask 1B, the absolute path to the file containing all "
                             "possible persuasion techniques",
                        type=str)

    args = parser.parse_args()
    techniques_file_path = args.techniques_file_path
    pred_files_path = args.pred_files_path
    subtask = args.subtask

    if validate_files(pred_files_path, subtask, techniques_file_path):
        techniques = []

        if subtask.startswith("1B"):
            techniques = read_techniques(techniques_file_path)

        for pred_file_path in pred_files_path:
            logging.info("Checking file: {}".format(pred_file_path))

            check_format(pred_file_path, subtask, techniques)
