

subtask_files = {
    "1A": {
        "train": "./../data/task1/task1A_train.jsonl",
        "dev": "./../data/task1/task1A_dev.jsonl",
        "test": "./../data/task1/task1A_test.jsonl"
    },
    "1B": {
        "train": "./../data/task1/task1B_train.jsonl",
        "dev": "./../data/task1/task1B_dev.jsonl",
        "test": "./../data/task1/task1B_test.jsonl"
    },
    "2A": {
        "train": "./../data/task2/ArAiEval23_disinfo_subtask2A_train.jsonl",
        "dev": "./../data/task2/ArAiEval23_disinfo_subtask2A_dev.jsonl",
        "test": "./../data/task2/ArAiEval23_disinfo_subtask2A_test.jsonl"
    },
    "2B": {
        "train": "./../data/task2/ArAiEval23_disinfo_subtask2B_train.jsonl",
        "dev": "./../data/task2/ArAiEval23_disinfo_subtask2B_dev.jsonl",
        "test": "./../data/task2/ArAiEval23_disinfo_subtask2B_test.jsonl"
    }   
}

id2label = {
    "1A": {0: "false", 1: "true"},
    "1B": {0: 'no_technique', 1: 'Loaded_Language', 2: 'Name_Calling-Labeling', 3: 'Questioning_the_Reputation', 4: 'Appeal_to_Fear-Prejudice', 5: 'Doubt', 6: 'Obfuscation-Vagueness-Confusion', 7: 'Causal_Oversimplification', 8: 'Exaggeration-Minimisation', 9: 'Slogans', 10: 'Appeal_to_Values', 11: 'Flag_Waving', 12: 'Red_Herring', 13: 'Appeal_to_Authority', 14: 'Straw_Man', 15: 'Conversation_Killer', 16: 'Repetition', 17: 'Appeal_to_Hypocrisy', 18: 'False_Dilemma-No_Choice', 19: 'Guilt_by_Association', 20: 'Appeal_to_Time', 21: 'Consequential_Oversimplification', 22: 'Whataboutism', 23: 'Appeal_to_Popularity'},
    "2A": {0: "no-disinfo", 1: "disinfo"},
    "2B": {0: "HS", 1: "OFF", 2:"SPAM", 3:"Rumor"},
}

label2id = {
    "1A": {"false": 0, "true": 1},
    "1B": {'no_technique': 0, 'Loaded_Language': 1, 'Name_Calling-Labeling': 2, 'Questioning_the_Reputation': 3, 'Appeal_to_Fear-Prejudice': 4, 'Doubt': 5, 'Obfuscation-Vagueness-Confusion': 6, 'Causal_Oversimplification': 7, 'Exaggeration-Minimisation': 8, 'Slogans': 9, 'Appeal_to_Values': 10, 'Flag_Waving': 11, 'Red_Herring': 12, 'Appeal_to_Authority': 13, 'Straw_Man': 14, 'Conversation_Killer': 15, 'Repetition': 16, 'Appeal_to_Hypocrisy': 17, 'False_Dilemma-No_Choice': 18, 'Guilt_by_Association': 19, 'Appeal_to_Time': 20, 'Consequential_Oversimplification': 21, 'Whataboutism': 22, 'Appeal_to_Popularity': 23},
    "2A": {"no-disinfo": 0, "disinfo": 1},
    "2B": {"HS": 0, "OFF": 1, "SPAM":2, "Rumor":3},
}

# Pre-computed using the formula here: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
class_weights = {
    "1A": [2.3841, 0.6327],
    "2A": [2.6673, 0.6154],
    "2B": [0.4378, 1.3240, 3.4842, 1.4843]
}