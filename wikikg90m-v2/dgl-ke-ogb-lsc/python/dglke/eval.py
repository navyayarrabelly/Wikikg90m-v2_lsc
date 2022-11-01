# -*- coding: utf-8 -*-
#
# eval.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#

import argparse
import os
import logging
import time
import pickle
import torch
from utils import get_compatible_batch_size, CommonArgParser_eval

from dataloader import EvalDataset, TrainDataset
from dataloader import get_dataset
from save_test_submission import get_test_predictions

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import multiprocessing as mp
    from .train_mxnet import load_model_from_checkpoint
    from .train_mxnet import test
else:
    import torch.multiprocessing as mp
    from train_pytorch import load_model_from_checkpoint
    from train_pytorch import test, test_mp

class ArgParser(CommonArgParser_eval):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE',
                                   'SimplE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge '\
                                  'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '\
                                  'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs,'\
                                  'the foramt should be built_in. For users own knowledge graphs,'\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE'\
                                  'on their own datasets. If the format is raw_udd_{htr},'\
                                  'users need to provide train_file [valid_file] [test_file].'\
                                  'If the format is udd_{htr}, users need to provide'\
                                  'entity_file relation_file train_file [valid_file] [test_file].'\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--model_path', type=str, default='ckpts',
                          help='The path of the directory where models are saved.')
        self.add_argument('--save_path', type=str, default='ckpts',
                          help='The path of the directory where ouput predictions are saved.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The negative sampling size for evaluation.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Negative sampling proportional to vertex degree for evaluation.')
        self.add_argument('--hidden_dim', type=int, default=256,
                          help='The hidden dim used by relation and entity')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('--LRE_rank',  type=int, default=200,
                          help='rank for low dimensional reduction')
        self.add_argument('--LRE', action='store_true',
                          help='rank for low dimensional reduction')
            
        
        self.add_argument('--eval_percent', type=float, default=1,
                          help='The percentage of data used for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Evaluate a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entitiy dim for complex number It is used by RotatE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to evaluate the model in parallel.'\
                                  'For multi-GPU, the number of processes by default is set to match the number of GPUs.'\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to evaluate the model in each process.'\
                                  'This argument is used for multiprocessing computation.')
        self.add_argument('--loss_genre', default='Logsigmoid',
                          choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE'],
                          help='The loss function used to train KGEM.')
        self.add_argument('--print_on_screen', action='store_true')
        self.add_argument('--encoder_model_name', type=str, default='emb',
                          help='emb or roberta or both')
        self.add_argument('--model_prefix', type=str, default='',
                          help='prefix for model foldr')

    def parse_args(self):
        args = super().parse_args()
        return args

def main():
    args = ArgParser().parse_args()
    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    assert os.path.exists(args.model_path), 'No existing model_path: {}'.format(args.model_path)

    assert args.dataset =='wikikg90m'
    args.neg_sample_size_eval = 1000

    # load dataset and samplers
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files, test_mode="test-dev")
    testchallenge_dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files, test_mode="test-challenge")
    args.train = False
    args.valid = True
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    args.has_edge_importance = False
    args.model_path= os.path.join(args.model_path, args.model_prefix)
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0, \
                'The number of processes needs to be divisible by the number of GPUs'

    # Here we want to use the regualr negative sampler because we need to ensure that
    # all positive edges are excluded.
    eval_dataset = EvalDataset(dataset, args)
    test_challenge_eval_datatset= EvalDataset(testchallenge_dataset, args)

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = args.neg_sample_size = test_challenge_eval_datatset.g.number_of_nodes()
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

    args.num_workers = 8 # fix num_workers to 8
    if args.num_proc > 1:
        test_sampler_tails = []
        test_sampler_heads = []
        test_challenge_sampler_tails=[]
        for i in range(args.num_proc):
            # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
            #                                                 args.neg_sample_size_eval,
            #                                                 args.neg_sample_size_eval,
            #                                                 args.eval_filter,
            #                                                 mode='head',
            #                                                 num_workers=args.num_workers,
            #                                                 rank=i, ranks=args.num_proc)
            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='tail',
                                                            num_workers=args.num_workers,
                                                            rank=i, ranks=args.num_proc)
            #test_sampler_heads.append(test_sampler_head)
            test_sampler_tails.append(test_sampler_tail)
    else:
        # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
        #                                                 args.neg_sample_size_eval,
        #                                                 args.neg_sample_size_eval,
        #                                                 args.eval_filter,
        #                                                 mode='head',
        #                                                 num_workers=args.num_workers,
        #                                                 rank=0, ranks=1)
        test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.eval_filter,
                                                        mode='tail',
                                                        num_workers=args.num_workers,
                                                        rank=0, ranks=1)
        test_challenge_sampler_tail = test_challenge_eval_datatset.create_sampler('test', args.batch_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.eval_filter,
                                                        mode='tail',
                                                        num_workers=args.num_workers,
                                                        rank=0, ranks=1)
        
    if args.num_proc > 1:
        valid_sampler_tails = []
        valid_sampler_heads = []
        for i in range(args.num_proc):
            # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
            #                                                 args.neg_sample_size_eval,
            #                                                 args.neg_sample_size_eval,
            #                                                 args.eval_filter,
            #                                                 mode='head',
            #                                                 num_workers=args.num_workers,
            #                                                 rank=i, ranks=args.num_proc)
            valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.neg_sample_size_eval,
                                                            args.eval_filter,
                                                            mode='tail',
                                                            num_workers=args.num_workers,
                                                            rank=i, ranks=args.num_proc)
            #test_sampler_heads.append(test_sampler_head)
            valid_sampler_tails.append(valid_sampler_tail)
    else:
        # test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
        #                                                 args.neg_sample_size_eval,
        #                                                 args.neg_sample_size_eval,
        #                                                 args.eval_filter,
        #                                                 mode='head',
        #                                                 num_workers=args.num_workers,
        #                                                 rank=0, ranks=1)
        valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.neg_sample_size_eval,
                                                        args.eval_filter,
                                                        mode='tail',
                                                        num_workers=args.num_workers,
                                                        rank=0, ranks=1)

    # load model
    n_entities = dataset.n_entities
    n_relations = dataset.n_relations
    ckpt_path = args.model_path
    args.num_proc =1
    model = load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path, dataset.entity_feat.shape[1], dataset.relation_feat.shape[1])
    if args.encoder_model_name in ['roberta', 'concat']:
        model.entity_feat.emb = dataset.entity_feat
        model.relation_feat.emb = dataset.relation_feat

    if args.num_proc > 1:
        model.share_memory()
    # test
    args.step = 0
    args.max_step = 0
    start = time.time()
    if args.num_proc > 1:
        queue = mp.Queue(args.num_proc)
        procs = []
        for i in range(args.num_proc):
            proc = mp.Process(target=test_mp, args=(args,
                                                    model,
                                                    [test_sampler_heads[i], test_sampler_tails[i]],
                                                    i,
                                                    'Test',
                                                    # queue
                                                    ))
            procs.append(proc)
            proc.start()

        # total_metrics = {}
        # metrics = {}
        # logs = []
        # for i in range(args.num_proc):
        #     log = queue.get()
        #     logs = logs + log

        # for metric in logs[0].keys():
        #     metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        # print("-------------- Test result --------------")
        # for k, v in metrics.items():
        #     print('Test average {}: {}'.format(k, v))
        # print("-----------------------------------------")

        for proc in procs:
            proc.join()
    else:
        test_input_dict = test(args, model,  [test_sampler_tail],step=0,rank=0,mode='Test')
        torch.save(test_input_dict, os.path.join(
                    args.save_path, "test_0_final.pkl"))
        valid_input_dict = test(args, model,  [valid_sampler_tail],step=0,rank=0,mode='valid')
        torch.save(valid_input_dict, os.path.join(
                    args.save_path, "valid_0_final.pkl"))
        test_challenge_input_dict = test(args, model,  [test_challenge_sampler_tail],step=0,rank=0,mode='Test')
        torch.save(test_challenge_input_dict, os.path.join(
                    args.save_path, "test_challenge_0_final.pkl"))
        print("saving in path "+ args.save_path)
    print('Test takes {:.3f} seconds'.format(time.time() - start))


    # parameters for evaluating and generating the test predictions.
    path = args.save_path
    valid_candidate_path =os.path.join(args.data_path, "wikikg90m-v2/processed/val_t_candidate.npy")
    test_candidate_path =os.path.join(args.data_path, "wikikg90m-v2/processed/test_dev_t_candidate_60000.npy")

    get_test_predictions(path,valid_candidate_path,test_candidate_path,mode="test-dev",num_proc=args.num_proc)


if __name__ == '__main__':
    main()
