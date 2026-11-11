#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import ast
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from data_set import DataSet
from model import ECRD
from trainer import Trainer


def parse_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise argparse.ArgumentTypeError('Use a list such as [10, 50].') from error
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise argparse.ArgumentTypeError('Provide a nonempty list.')
    return list(parsed)


def parse_bool(value):
    value = value.lower()
    if value in ('true', '1'):
        return True
    if value in ('false', '0'):
        return False
    raise argparse.ArgumentTypeError('Use true or false.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ECRD Model Configuration')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension (Sec. V-A3)')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='Squared L2 weight alpha (Eq. 13)')
    parser.add_argument('--layers', type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument('--node_dropout', type=float, default=0.0, help='Optional encoder dropout probability')
    parser.add_argument('--message_dropout', type=float, default=0.25, help=argparse.SUPPRESS)
    parser.add_argument('--dim_qk', type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument('--dim_v', type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument('--omega', type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument('--data_name', choices=['beibei', 'taobao', 'tmall', 'jdata', 'tenrec'], default='taobao')
    parser.add_argument('--behaviors', action='append', help=argparse.SUPPRESS)
    parser.add_argument('--loss_type', type=str, default='bpr', help='Training sample format')
    parser.add_argument('--num_hops', type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument('--gamma', type=float, default=0.3, help='Initial-space cosine threshold (Sec. IV-B2)')
    parser.add_argument('--if_load_model', type=parse_bool, nargs='?', const=True, default=False)
    parser.add_argument('--topk', type=parse_list, default=[10, 30, 50, 100, 200])
    parser.add_argument('--metrics', type=parse_list, default=['hit', 'ndcg'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size (Sec. V-A3)')
    parser.add_argument('--test_batch_size', type=int, default=256, help='Validation and test batch size')
    parser.add_argument('--min_epoch', type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--model_path', default='./check_point', help='Checkpoint directory')
    parser.add_argument('--check_point', default='ecrd_base.pth', help='Checkpoint file name')
    parser.add_argument('--model_name', default='ECRD', help='Model identifier')
    parser.add_argument('--pt_loop', type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--tp', default='ECRD_exp', help='Results directory name')
    parser.add_argument('--layers_nums', type=parse_list, help='Propagation depth for each behavior')
    parser.add_argument('--loss_coefficient', type=parse_list, help='BPR weight for each behavior')
    parser.add_argument('--tau', type=float, default=0.2, help='DCC temperature (Eq. 12)')
    parser.add_argument('--alpha', type=float, help='DCC weight lambda (Eq. 13)')
    parser.add_argument('--k_semantic', type=int, help='Semantic candidate count K (Eq. 4)')
    args = parser.parse_args()

    # Sec. V-A1: The target behavior is last in the cascade.
    args.data_path = f'./data/{args.data_name}/'
    args.behaviors = ['view', 'cart', 'buy']
    if args.data_name in ('tmall', 'jdata', 'tenrec'):
        args.behaviors.insert(1, 'collect')
    behavior_count = len(args.behaviors)
    if args.layers_nums is not None and len(args.layers_nums) != behavior_count:
        logger.warning('Using the default cascade depths for the {} behaviors.', behavior_count)
        args.layers_nums = None
    if args.layers_nums is None:
        args.layers_nums = [1] * (behavior_count - 1) + [3]
    if args.loss_coefficient is None:
        args.loss_coefficient = [1] * behavior_count
    if len(args.loss_coefficient) != behavior_count:
        parser.error('Provide one BPR weight per behavior.')
    if any(type(depth) is not int or depth < 0 for depth in args.layers_nums):
        parser.error('--layers_nums must contain nonnegative integers.')
    if any(type(k) is not int or k < 1 for k in args.topk):
        parser.error('--topk must contain positive integers.')
    ranking_metrics = {'hit', 'ndcg', 'precision', 'map', 'recall', 'mrr'}
    if any(not isinstance(metric, str) or metric.lower() not in ranking_metrics for metric in args.metrics):
        parser.error('--metrics must contain ranking metrics such as ["hit", "ndcg"].')
    if min(args.batch_size, args.test_batch_size, args.epochs) < 1:
        parser.error('Batch sizes and epochs must be positive.')
    if args.tau <= 0:
        parser.error('--tau must be positive.')
    if args.alpha is None:
        args.alpha = 0.1 if args.data_name == 'beibei' else 0.3
    if args.k_semantic is None:
        args.k_semantic = 1 if args.data_name == 'beibei' else 2
    if args.k_semantic < 0:
        parser.error('--k_semantic must be nonnegative.')
    if args.decay:
        logger.warning('Adam weight decay is disabled. Use --reg_weight for Eq. (13).')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    dataset = DataSet(args)
    model = ECRD(args, dataset)
    trainer = Trainer(model, dataset, args)
    logger.info(args)
    logger.info(model)
    trainer.train_model()

    # Sec. V-A3: Test the selected checkpoint once.
    result = trainer.evaluate(
        trainer.best_epoch - 1, args.test_batch_size, dataset.test_dataset(),
        dataset.test_interacts, dataset.test_gt_length, mask_validation=True,
    )
    logger.info('Test metrics: {}', result)

    results_path = Path('results') / args.tp / f'{args.data_name}.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = ['config', 'seed', *result]
    rows = []
    if results_path.is_file():
        with results_path.open(newline='', encoding='utf-8') as file:
            rows = list(csv.reader(file))
        if rows and rows[0] != header:
            raise ValueError('The results CSV has different metric columns. Use a new --tp value.')
    config_id = len(rows) if rows else 1
    config_path = Path('conf') / args.data_name / f'{config_id}.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open('w', encoding='utf-8') as file:
        json.dump({**vars(args), 'best_epoch': trainer.best_epoch}, file)
    with results_path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not rows:
            writer.writerow(header)
        writer.writerow([config_id, args.seed, *(round(score, 4) for score in result.values())])
