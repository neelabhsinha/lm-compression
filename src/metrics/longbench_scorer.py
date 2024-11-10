from src.metrics.metrics import *


class LongBenchEvaluationMetric:
    def __init__(self):
        self.dataset2metric = {
            "hotpotqa": qa_f1_score,
            "2wikimqa": qa_f1_score,
            "musique": qa_f1_score,
            "narrativeqa": qa_f1_score,
            "qasper": qa_f1_score,
            "multifieldqa_en": qa_f1_score,
            "gov_report": rouge_score,
            "qmsum": rouge_score,
            "trec": classification_score,
            "nq": qa_f1_score,
            "triviaqa": qa_f1_score,
            "passage_retrieval_en": retrieval_score,
            "passage_count": count_score,
            "lcc": code_sim_score,
            "repobench-p": code_sim_score,
            "multi_news": rouge_score,
            "samsum": rouge_score
        }

    def get_score(self, dataset, prediction, answers, all_classes):
        score = 0.
        for ground_truth in answers:
            score = max(score, self.dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        score = score * 100
        return score
