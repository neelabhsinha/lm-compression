import re
import string

from fuzzywuzzy import fuzz
import difflib

from typing import List
from collections import Counter
from rouge import Rouge


def normalize_answer(s):  # used by other functions
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):  # Accuracy (EM) as per paper
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    prediction = re.sub(r'/\*.*?\*/', '', prediction, flags=re.DOTALL)
    prediction = re.sub(r'(""".*?""")|(\'\'\'.*?\'\'\')', '', prediction, flags=re.DOTALL)
    cleaned_lines = []
    for line in prediction.split('\n'):
        line = line.split('#')[0].split('//')[0].strip()
        if line:
            cleaned_lines.append(line)
    cleaned_prediction = '\n'.join(cleaned_lines)
    cleaned_prediction = re.sub(r'\s+', ' ', cleaned_prediction).strip()
    normalized_ground_truth = re.sub(r'\s+', ' ', ground_truth).strip()
    return fuzz.ratio(cleaned_prediction, normalized_ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):  # Accuracy (CLS) as per paper
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = (1.0 / len(em_match_list))
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):  # ROUGE-L as per paper
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]


def f1_score(prediction, ground_truth, **kwargs):  # used by qa_f1_score (internal use)
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = recall
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):  # F1 (as per paper)
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)
