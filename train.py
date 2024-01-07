#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model for question answering using 🤗 Accelerate.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
from multiprocessing import context
import os
import random
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import copy 
import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from utils import postprocess_qa_predictions
from perturb import (
    perturb, 
    produce_no_answer_batch, 
    batch_get_answer_tokens,
    batch_get_answer_tokens_topk,
    batch_compute_mIoU,
    get_topk,
    flatten_column,
)
from sklearn.metrics import f1_score
import pdb 
import pandas as pd 
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def save_prefixed_metrics(results, output_dir, file_name: str = "all_results.json", metric_key_prefix: str = "eval"):
    """
    Save results while prefixing metric names.

    Args:
        results: (:obj:`dict`):
            A dictionary of results.
        output_dir: (:obj:`str`):
            An output directory.
        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):
            An output file name.
        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):
            A metric name prefix.
    """
    # Prefix all keys with metric_key_prefix + '_'
    for key in list(results.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            results[f"{metric_key_prefix}_{key}"] = results.pop(key)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(results, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--preprocessing_num_workers", type=int, default=4, help="A csv or a json file containing the training data."
    )
    parser.add_argument("--do_predict", action="store_true", help="To do prediction on the question answering model")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--custom_warmup_steps", type=int, default=0, help="Number of steps for the warmup before doing perturbation/permutation."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="The total number of n-best predictions to generate when looking for an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help=(
            "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        ),
    )
    parser.add_argument(
        "--no_answer_probability_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--use_threshold", action="store_true"
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, some of the examples do not have an answer.",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_predict_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of prediction examples to this",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--num_perturbation_examples_per_batch",
        type=int,
        default=0,
        help="Number of perturbation example to use per batch",
    )

    parser.add_argument(
        "--num_permutation_examples_per_batch",
        type=int,
        default=0,
        help="Number of permutation example to use per batch",
    )

    parser.add_argument(
        "--IoU_threshold",
        type=float,
        default=0.25, 
        help="Threshold of IoU to determine if an answer candidate is successfully perturbed"
    )

    parser.add_argument(
        "--num_retrieval",
        type=int,
        default=0,
        help="number of retrieval examples to use"
    )

    parser.add_argument(
        "--weight_perturb",
        type=float, default=1.0
    )

    parser.add_argument(
        "--weight_permute",
        type=float, default=1.0
    )

    parser.add_argument(
        "--weight_retrieval",
        type=float, default=1.0
    )

    parser.add_argument(
        "--use_paraphrase_detector",
        action="store_true", 
    )

    parser.add_argument(
        "--no_ans_only",
        action="store_true",
    )

    parser.add_argument(
        "--ans_only",
        action="store_true",
    )

    parser.add_argument(
        "--accept_everything_as_negative",
        action="store_true"
    )

    parser.add_argument(
        "--retrieval_data_path",
        type=str, default="./data/train-w-ret.csv"
    )

    parser.add_argument(
        "--retrieval_context_title_path",
        type=str, default="./data/train_documents.csv"
    )

    parser.add_argument("--remove_no_answer", action="store_true", help="force remove datapoint that has no answer")
    
    args = parser.parse_args()
    args.num_grad_accumulation_scale_up = 1 + args.num_perturbation_examples_per_batch + args.num_permutation_examples_per_batch

    # Sanity checks
    if (
        args.dataset_name is None
        and args.train_file is None
        and args.validation_file is None
        and args.test_file is None
    ):
        raise ValueError("Need either a dataset name or a training/validation/test file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_qa_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field="data")
    
    # Generate NoAns examples by retrieval
    if args.num_retrieval > 0:
        print(f"loading retrieval data from {args.retrieval_data_path}")
        df = pd.read_csv(args.retrieval_data_path)
        retrieval_column_names = df.columns[:-1]
        df_index_context_title = pd.read_csv(args.retrieve_context_title_path)
        index_to_context = {i : c for (i,c) in enumerate(df_index_context_title['context'].tolist())}
        context_to_index = {c: i for i, c in index_to_context.items()}
        index_to_title = {i : t for (i,t) in enumerate(df_index_context_title['title'].tolist())}
        num_rows = len(df)
        df['context_id'] = df.apply(lambda x: context_to_index[x.context], axis=1)
        df['retrieved_psgs_new'] = df.apply(lambda x: get_topk(x.retrieved_psgs, x.context_id, num_rows, topk=args.num_retrieval), axis=1)
        df_new = flatten_column(df, 'retrieved_psgs_new')
        df_new['context_original'] = df_new['context']
        df_new['title_original'] = df_new['title']
        df_new['context'] = df_new.apply(lambda x: index_to_context[x.retrieved_psgs_new], axis=1)
        df_new['title'] = df_new.apply(lambda x: index_to_title[x.retrieved_psgs_new], axis=1)
        df_new['answers'] = [{'text': [], 'answer_start': []} for _ in range(len(df_new))]
        df_new = df_new[retrieval_column_names]
        retrieval_dataset = datasets.Dataset.from_pandas(df_new)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForQuestionAnswering.from_config(config)
    
    tok_gen = BartTokenizer.from_pretrained("facebook/bart-large")
    generator = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)

    paraphrase_tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/roberta-base-qqp")
    paraphrase_classifier = AutoModelForSequenceClassification.from_pretrained("JeremiahZ/roberta-base-qqp")
    
    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # Training preprocessing
    def prepare_train_features_with_overflow(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["has_answer"] = []
        tokenized_examples["is_v1_example"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["is_v1_example"].append(False)
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["has_answer"].append(False)
            else:
                tokenized_examples["is_v1_example"].append(True)
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["has_answer"].append(False)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["has_answer"].append(True)
                
        return tokenized_examples

    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        #sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["has_answer"] = []
        tokenized_examples["is_v1_example"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["is_v1_example"].append(False)
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["has_answer"].append(False)
            else:
                tokenized_examples["is_v1_example"].append(True)
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["has_answer"].append(False)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["has_answer"].append(True)
                
        return tokenized_examples

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if args.max_train_samples is not None:
        # We will select sample from whole data if agument is specified
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # Create train feature from dataset
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )
        if args.num_retrieval > 0:
            retrieval_dataset = retrieval_dataset.map(
                prepare_train_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on retrieval dataset",
            )
        if args.remove_no_answer:
            train_dataset = train_dataset.filter(
                lambda example:
                    (not example['has_answer'] and not example['is_v1_example']) or
                    (example['has_answer'] and example['is_v1_example'])
            ) #XNOR
        train_dataset = train_dataset.remove_columns(['has_answer', 'is_v1_example'])
        if args.num_retrieval > 0:
            retrieval_dataset = retrieval_dataset.remove_columns(['has_answer', 'is_v1_example'])
        if args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
        
    if args.version_2_with_negative:
        logger.info("using v2 examples for validation")
        eval_examples = load_dataset("squad_v2", args.dataset_config_name)["validation"]
    else:
        eval_examples = raw_datasets["validation"] 
    
    if args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    # Validation Feature Creation
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if args.max_eval_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    if args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = raw_datasets["test"]
        if args.max_predict_samples is not None:
            # We will select sample from whole data
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        # Predict Feature Creation
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(
                prepare_validation_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            if args.max_predict_samples is not None:
                # During Feature creation dataset samples might increase, we will select required samples again
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, drop_last=True,
    )

    if args.num_retrieval > 0:
        retrieval_dataloader = DataLoader(
            retrieval_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, drop_last=True,
        )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(["example_id", "offset_mapping"])
        predict_dataloader = DataLoader(
            predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=args.version_2_with_negative,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
            #null_score_diff_threshold=args.null_score_diff_threshold,
            no_answer_probability_threshold=args.no_answer_probability_threshold,
            without_threshold=(not args.use_threshold), 
            output_dir=args.output_dir,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": prob} for k, (v, prob) in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, (v, _) in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = evaluate.load("squad_v2" if args.version_2_with_negative else "squad")

    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat
            
            
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_grouped_parameters_gen = [
        {
            "params": [p for n, p in generator.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in generator.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    optim_gen = torch.optim.AdamW(optimizer_grouped_parameters_gen, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    if args.num_retrieval > 0:
        retrieval_dataloader = accelerator.prepare(
            retrieval_dataloader
        )

    generator, optim_gen = accelerator.prepare(generator, optim_gen)
    paraphrase_classifier = accelerator.prepare(paraphrase_classifier)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("qa_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    
    generator.eval()
    paraphrase_classifier.eval()
    model.train()
    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.with_tracking:
            total_loss = 0
            total_gen_loss = 0
        
        if args.num_retrieval > 0:
            retrieval_dataloader_iterable = iter(retrieval_dataloader)
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue
            
            ###### compute model outputs for both original batch and perturb batch #####
            model.eval()
            no_pert_and_perm = (step <= args.custom_warmup_steps and epoch == 0)
            perturbation_info = []
            with torch.no_grad():
                outputs = model(**batch)
                start_logits, end_logits = outputs.start_logits, outputs.end_logits
                model_answer_tokens_topk = batch_get_answer_tokens_topk(tokenizer, start_logits, end_logits, batch['input_ids'], args)
                model_answer_tokens = [pred[0]['tokens'] for pred in model_answer_tokens_topk]
                m_answers = tokenizer.batch_decode(model_answer_tokens) # model predictions on original examples

                gt_answer_tokens = batch_get_answer_tokens(batch['start_positions'], batch['end_positions'], batch['input_ids'], args)
                gt_answers = tokenizer.batch_decode(gt_answer_tokens)

                for pt_idx in range(args.num_perturbation_examples_per_batch):
                    perturbed_batch, info, success_perturb, mask = \
                        perturb(batch, tokenizer, tok_gen, generator, paraphrase_tokenizer, paraphrase_classifier,\
                            args, max_seq_length, pad_on_right, accelerator.num_processes)
                    if not args.use_paraphrase_detector:
                        mask = torch.ones_like(mask) 
                    p_outputs = model(**perturbed_batch) # Model prediction on perturbed examples
                    p_start_logits, p_end_logits = p_outputs.start_logits, p_outputs.end_logits
                    p_answer_tokens_topk = batch_get_answer_tokens_topk(tokenizer, p_start_logits, p_end_logits, perturbed_batch['input_ids'], args)
                    batch_mIoU = batch_compute_mIoU(model_answer_tokens, p_answer_tokens_topk, args, logger)
                    p_start_positions = torch.zeros(args.per_device_train_batch_size).type(torch.LongTensor).to(model.device)
                    p_end_positions = torch.zeros(args.per_device_train_batch_size).type(torch.LongTensor).to(model.device)
                    for i in range(args.per_device_train_batch_size):
                        example_info = info[i]
                        m_answer = m_answers[i]  # model prediction on original example
                        g_answer = gt_answers[i] # groundtruth answer

                        pred_topk = p_answer_tokens_topk[i] 
                        pred_tokens_topk = [pred_topk[j]['tokens'] for j in range(len(pred_topk))]
                        p_answers = tokenizer.batch_decode(pred_tokens_topk) # topk predictions on perturbed example
                        IoU_list = batch_mIoU[i] # IoU between topk predictions and model prediction on original example
                        
                        answer_idx = 0
                        if mask[i] == 0: # two sentences are paraphrase
                            logger.info("Perturbation IS a paraphrase")
                            p_answer = p_answers[answer_idx]

                            if (m_answer == g_answer and  # model predicted correctly 
                                g_answer == p_answer):  # perturbation didn't change the label 
                                logger.info("Robust example")
                                mask[i] = 1 # Robust example, will be kept for training

                        else:
                            exists_good_p = False
                            for j in range(len(pred_tokens_topk)):
                                p_answer = p_answers[j]
                                IoU = IoU_list[j]
                                if IoU < args.IoU_threshold:
                                    logger.info("Exists perturbation")
                                    exists_good_p = True
                                    answer_idx = j
                                    break 
                            p_answer = p_answers[answer_idx] # best perturbed answer (minimum IoU with model prediction on original example)
                            if not exists_good_p: mask[i] = 0 # Non-paraphrase pertubation didn't change answer
                        
                        p_start_positions[i] = pred_topk[answer_idx]['start_index']
                        p_end_positions[i] = pred_topk[answer_idx]['end_index']
                        
                        success_perturb_i = success_perturb and (example_info['perturbation'] != example_info['question'])
                        if (m_answer == g_answer and  # model predicted correctly 
                            g_answer == p_answer and  # perturbation didn't change the label 
                            success_perturb_i):         # perturbed question is a valid perturbation
                            logger.info("Answer didn't change w.r.t. successful perturbation")
                            mask[i] = 1

                        if (tokenizer.cls_token in p_answer and  # perturbed prediction is the same as model prediction 
                            tokenizer.cls_token in m_answer  and  # both perturbed and orginal predictions are NoAns
                            tokenizer.cls_token not in g_answer): # groundtruth has answer
                            logger.info("NoAns prediction for both orginal and perturbed. Disgard.")
                            mask[i] = 0
                        
                        if not success_perturb_i:
                            logger.info("Unsuccessful perturbation. Disgard.")
                            mask[i] = 0

                        do_backprop = mask[i] > 0.5 # convert mask to boolean. if True, this example will be used for training (via backprop)
                        logger.info(f"context:          {example_info['context']}")
                        logger.info(f"question:         {example_info['question']}")
                        logger.info(f"gt answer:        {g_answer}")
                        logger.info(f"model answer:     {m_answer}")
                        logger.info(f"masked_q:         {example_info['masked_q']}")
                        logger.info(f"perturbation:     {example_info['perturbation']}")
                        logger.info(f"all pert answers: {p_answers}")
                        logger.info(f"topk answer IoU:  {[round(iou, 2) for iou in batch_mIoU[i]]}")
                        logger.info(f"perturbed answer: {p_answer}")
                        logger.info(f"do backprop:      {do_backprop}")
                        logger.info(f"in warm up?:      {no_pert_and_perm}\n")
                    logger.info(f"mask: {mask}")
                    perturbation_info.append({
                        'perturbed_batch': perturbed_batch,
                        'p_start_positions': p_start_positions,
                        'p_end_positions': p_end_positions,
                        'mask': mask 
                    })
            
            model.train()
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                logger.info(f"model loss: {loss.detach().float()}")

                if not no_pert_and_perm:
                    # Perturbed cases
                    for pt_idx in range(args.num_perturbation_examples_per_batch):
                        perturbed_batch = perturbation_info[pt_idx]['perturbed_batch']
                        p_start_positions = perturbation_info[pt_idx]['p_start_positions']
                        p_end_positions = perturbation_info[pt_idx]['p_end_positions']
                        if args.no_ans_only: 
                            no_ans_mask = (p_start_positions == 0) * (p_end_positions == 0)
                            mask *= no_ans_mask
                        if args.ans_only:
                            ans_mask = torch.logical_not((p_start_positions == 0) * (p_end_positions == 0))
                            mask *= ans_mask
                        

                        mask = perturbation_info[pt_idx]['mask']
                        outputs = model(**perturbed_batch)
                        start_logits, end_logits = outputs.start_logits, outputs.end_logits
                        ignored_index = start_logits.size(1)
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, reduction='none')

                        if args.accept_everything_as_negative:
                            mask = torch.ones_like(mask)
                            p_start_positions = torch.zeros_like(p_start_positions)
                            p_end_positions = torch.zeros_like(p_end_positions)
                            
                        start_loss = loss_fct(start_logits, p_start_positions) * mask
                        end_loss = loss_fct(end_logits, p_end_positions) * mask
                        p_loss = (start_loss.sum() + end_loss.sum()) / (2 * mask.sum()) if  mask.sum() > 0 else 0.0 * (start_loss.sum() + end_loss.sum())
                        p_loss *= args.weight_perturb/args.num_perturbation_examples_per_batch
                        accelerator.backward(p_loss)
                        logger.info(f"perturbed [idx: {pt_idx}] loss: {p_loss.detach().float()}")
                        
                    # adding permutation
                    for pm_idx in range(args.num_permutation_examples_per_batch):
                        batch_perm, mask = produce_no_answer_batch(batch, tokenizer, args, max_seq_length, pad_on_right, logger)
                        outputs = model(**batch_perm)
                        
                        start_logits, end_logits = outputs.start_logits, outputs.end_logits
                        ignored_index = start_logits.size(1)
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
                        p_start_positions = batch_perm['start_positions']
                        p_end_positions = batch_perm['end_positions']
                        start_loss = loss_fct(start_logits, p_start_positions) * mask
                        end_loss = loss_fct(end_logits, p_end_positions) * mask
                        loss = (start_loss.sum() + end_loss.sum()) / (2 * mask.sum()) if  mask.sum() > 0 else 0.0 * (start_loss.sum() + end_loss.sum())
                        loss *= args.weight_permute/args.num_permutation_examples_per_batch
                        accelerator.backward(loss)

                        logger.info(f"perm [idx: {pm_idx}] loss: {loss.detach().float()}")

                    # adding retrieval-based no answerable question
                    for rt_idx in range(args.num_retrieval):
                        try:
                            batch_retv = next(retrieval_dataloader_iterable)
                        except StopIteration:
                            retrieval_dataloader_iterable = iter(retrieval_dataloader)
                            batch_retv = next(retrieval_dataloader_iterable)
                        outputs = model(**batch_retv)
                        rt_loss = outputs.loss
                        rt_loss *= args.weight_retrieval/args.num_retrieval
                        accelerator.backward(rt_loss)
                        logger.info(f"retrieval-based loss: {rt_loss.detach().float()}")

                # accumulate gradient and update the parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

    # Evaluation
    logger.info("***** Running Evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

    all_start_logits = []
    all_end_logits = []

    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

            all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
            all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    logger.info(f"Evaluation metrics: {eval_metric}")

    # Prediction
    if args.do_predict:
        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(predict_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        all_start_logits = []
        all_end_logits = []

        model.eval()

        for step, batch in enumerate(predict_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                    end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)

                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, predict_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, predict_dataset, max_len)

        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Predict metrics: {predict_metric}")

    if args.with_tracking:
        log = {
            "squad_v2" if args.version_2_with_negative else "squad": eval_metric,
            "train_loss": total_loss.item() / len(train_dataloader),
            "generator_loss": total_gen_loss.item() / len(train_dataloader),
            "epoch": epoch,
            "step": completed_steps,
        }
    if args.do_predict:
        log["squad_v2_predict" if args.version_2_with_negative else "squad_predict"] = predict_metric

        accelerator.log(log, step=completed_steps)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            logger.info(json.dumps(eval_metric, indent=4))
            save_prefixed_metrics(eval_metric, args.output_dir)


if __name__ == "__main__":
    main()
