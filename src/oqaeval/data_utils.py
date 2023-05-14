import csv
import json
import logging
import os
import regex
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Sequence, Tuple, Union

import datasets

logger = logging.getLogger("data_utils")


class SimpleTokenizer:
    """
    Inspired by https://github.com/castorini/pyserini/blob/pyserini-0.21.0/pyserini/eval/evaluate_dpr_retrieval.py#L165
    """

    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            f"({self.ALPHA_NUM})|({self.NON_WS})", flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def _amend(self, text: str) -> str:
        text = text.replace("\u2019", "'").replace("--", "_")

        for special_word in ("don't", "ain't", "won't", "didn't", "can't", "doesn't", "wanna", "gonna", "gimme"):
            if special_word in text:
                if special_word.endswith("nna"):
                    fixed = f"{special_word[:-len('na')]} na"
                elif special_word.endswith("mme"):
                    fixed = f"{special_word[:-len('me')]} me"
                else:
                    fixed = special_word[: -len("n't")] + " n't"

                text = text.replace(special_word, fixed)

        return text

    def tokenize(self, text: str, as_string: bool = False) -> Union[str, List[str]]:
        text = self._amend(text)

        tokens = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            tokens.append(matches[i].group())

        if as_string:
            return " ".join(tokens)
        else:
            return tokens


@dataclass
class Question:
    text: str
    answers: Union[Set[str], List[str]]
    id: Optional[str] = None
    tokens: Optional[List[str]] = field(default=None)
    acceptable_answers: Optional[List[str]] = field(default=None)
    unacceptable_answers: Optional[List[str]] = field(default=None)

    @property
    def has_answers(self) -> bool:
        return self.answers and len(self.answers) > 0
    
    @property
    def has_annotated_answers(self) -> bool:
        return len(self.gold_answers) > 0 or self.unacceptable_answers

    @property
    def tokenized_text(self) -> Optional[str]:
        return " ".join(self.tokens) if self.tokens is not None else None

    def update_answers(self, annotated_answers):
        if not annotated_answers:
            return

        self.acceptable_answers = annotated_answers["yes"]
        self.unacceptable_answers = annotated_answers["no"]

    def is_unacceptable(self, candidate_answer: str) -> bool:
        if self.unacceptable_answers:
            for ans in self.unacceptable_answers:
                if candidate_answer == ans or candidate_answer.lower() == ans.lower():
                    return True

        return False

    @property
    def gold_answers(self) -> Set[str]:
        answers = set(self.answers) if self.answers else set()
        
        if self.acceptable_answers:
            answers.update(self.acceptable_answers)

        if self.unacceptable_answers:
            for a in self.unacceptable_answers:
                if a in answers:
                    answers.remove(a)
                elif a.lower() in answers:
                    answers.remove(a.lower())
                    
        return answers

    def to_json(self) -> Dict[str, Any]:
        json_dict = dict(
            question=self.text,
            id=self.id,
            answers=self.answers,
        )

        return json_dict

    @classmethod
    def from_json(cls, q_dict, idx: int = 0):
        return Question(
            q_dict["question"],
            q_dict.get("answer", q_dict.get("answers", None)),
            q_dict.get("id", idx),
        )


def read_json(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as reader:
        dataset = json.load(reader)
    for i, q in enumerate(dataset):
        tokens = tokenizer.tokenize(q["question"])
        yield Question(q["question"], q["answers"], str(i), tokens=tokens)


def reader_tsv(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(dataset_reader):
            tokens = tokenizer.tokenize(row[0].strip())

            try:
                answers = [a.strip() for a in eval(row[1])]
            except:
                answers = row[1].strip().split(" | ")

            yield Question(row[0].strip(), answers, str(i), tokens=tokens)


def read_jsonl(dataset_path: os.PathLike) -> Iterable[Question]:
    tokenizer = SimpleTokenizer()

    with open(dataset_path, "r", encoding="utf-8") as dataset_reader:
        for i, line in enumerate(dataset_reader):
            if not line.strip():
                continue

            q_dict = json.loads(line)
            yield Question(
                q_dict["question"],
                q_dict.get("answer", q_dict.get("answers", None)),
                q_dict.get("id", str(i)),
                tokenizer.tokenize(q_dict["question"]),
            )


def load_dataset(dataset_name: str, split: str = "validation"):
    tokenizer = SimpleTokenizer()
    for i, ex in enumerate(datasets.load_dataset(dataset_name, split=split)):
        answers = ex["answer"]
        if isinstance(answers, dict):
            # TriviaQA
            answers = [answers["value"]] + [answers["aliases"]]
        yield Question(
            ex["question"],
            answers,
            ex.get("question_id", str(i)),
            tokenizer.tokenize(ex["question"]),
        )


def read_questions(dataset_file: os.PathLike, split: str = "validation") -> Iterable[Question]:
    dataset_path = Path(dataset_file)
    if dataset_path.exists():
        if dataset_path.suffix == ".json":
            return list(read_json(dataset_path))
        elif dataset_path.suffix in (".tsv", ".txt"):
            return list(reader_tsv(dataset_path))
        elif dataset_path.suffix == ".jsonl":
            return list(read_jsonl(dataset_path))
        else:
            raise ValueError(f"Unknown file format: {dataset_path.suffix}")
    else:
        return list(datasets.load_dataset(dataset_file, split))


def read_predict_file(
    predict_file: os.PathLike,
    questions: Optional[Sequence[Question]] = None,
) -> Tuple[Mapping[str, str], Sequence[Question]]:
    """
    Loading predictions from file

    :param predict_file: Path a file containing predictions
    :param questions: a sequence of questions in the dataset
    :return: a dict mapping questions to predicted answers

    Supported file formats are json, jsonl, tsv, and csv

    json format:
    [
        {"question": "when was the first hunger games book published", "prediction": "September 14, 2008"},
        ...
    ]

    jsonl format:
    {"question": "when was the first hunger games book published", "prediction": "September 14, 2008"}

    tsv/csv format (should be aligned with questions from dataset file):
    1       May 18, 2018
    """
    tokenizer = SimpleTokenizer()
    predict_file = Path(predict_file)

    collected_questions = []
    predicted_dict = {}
    if predict_file.suffix in (".tsv", ".csv", ".txt"):
        assert questions, "dataset file must be provided"

        with predict_file.open("r") as p:
            reader = csv.reader(p, delimiter="," if predict_file.suffix == ".csv" else "\t")

            for predicted_row, q in zip(reader, questions):
                predicted_dict[q.tokenized_text.lower()] = predicted_row[1].strip()
    else:
        if predict_file.suffix == ".json":
            with predict_file.open("r") as p:
                predictions = json.load(p)
        else:
            with predict_file.open("r") as p:
                predictions = [json.loads(line) for line in p if line.strip()]

        answer_dict = {}
        for predicted_item in predictions:
            predicted_method = predicted_item["prediction"]
            if isinstance(predicted_method, (list, tuple)):
                predicted_answer = predicted_method[0].strip()
            else:
                predicted_answer = predicted_method.strip()

            q = Question(
                predicted_item["question"],
                predicted_item["answer"],
                tokens=tokenizer.tokenize(predicted_item["question"]),
            )
            if not questions:
                collected_questions.append(q)
            answer_dict[q.tokenized_text.lower()] = predicted_answer

        if questions:
            for q in questions:
                if q.tokenized_text.lower() in answer_dict:
                    predicted_dict[q.tokenized_text.lower()] = answer_dict[q.tokenized_text.lower()]
        else:
            questions = collected_questions
            predicted_dict = answer_dict

    return predicted_dict, questions


def read_annotations(annotation_file: os.PathLike) -> Mapping[str, Mapping[str, Set[str]]]:
    """
    Reads the content of an annotation file
    :param annotation_file: Path to annotation file
    :return: a dictionary with questions as keys mapped to two sets of acceptable and unacceptable answers

    Annotation file expected to be a tsv file with the following format:
    Question        Model answer    Acceptable?
    when did earth's atmosphere change due to living organisms  2.4 billion years ago    Yes
    """
    tokenizer = SimpleTokenizer()
    annotated_answers = defaultdict(lambda: {"yes": set(), "no": set()})

    with open(annotation_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            question = tokenizer.tokenize(row["Question"], as_string=True).lower()
            if "Acceptable?" not in row or not row["Acceptable?"].strip():
                continue

            annotated_answers[question][row["Acceptable?"].strip().lower()].add(row["Model answer"])
            if question.endswith("?"):
                annotated_answers[question[:-1].strip()][row["Acceptable?"].strip().lower()].add(row["Model answer"])

    logger.info(f"Annotations loaded with {len(annotated_answers)} entries")
    return dict(annotated_answers)