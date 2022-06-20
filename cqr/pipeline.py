import sys
sys.path.append('/home/admin/baselines/Ours')
from cqr.train_model import TrainModel
from cqr.inference_model import InferenceModel
import copy
import argparse
import logging
import json
import os
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import tqdm, trange
from transformers import  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from cqr.contrastive_loss import ContrastiveLoss
from cqr.dataset import QueryRewriteDataset, SimplifierDataset
from cqr.utils import NUM_FOLD, set_seed, special_tokens_dict
logger = logging.getLogger(__name__)
import pickle
def predict(input_list_raw,inference_model,batch_size=32,threshold=80.):
    output_list = []
    input_list = copy.deepcopy(input_list_raw)
    for i in tqdm(range(0,len(input_list),batch_size),desc="Predict"):
        input_sents = [s['input'] for s in input_list[i:i+batch_size]]
        prediction,confs = inference_model.predict(input_sents)
#         print(confs)
        for m,conf in enumerate(confs):
            if conf>=threshold:
                record = input_list[i+m]
                record['target'] = prediction[m]
                output_list.append(record)
#         print(confs[0])
    return output_list
def data_removal(a,b):
    c = copy.deepcopy(b)
    for i in c:
        del i['target']
    return [i for i in a if i not in c]
def exchange(a):
    for data in a:
            inputs = copy.deepcopy(data['input'])
            data['input'][-1] = data['target']
            data['target'] = inputs[-1]
    return a
    
def pipeline(args,train_gold_files,unlabeled_file_s,unlabeled_file_r,model_s,model_r,tokenizer,logger,i):       
    unlabeled_data_s = [json.loads(i.strip('\n')) for i in open(unlabeled_file_s).readlines()]
    unlabeled_data_r = [json.loads(i.strip('\n')) for i in open(unlabeled_file_r).readlines()]
    labeled_data = []
    num = 1
#     if not args.zero_shot:
    for t in train_gold_files:
        labeled_data.extend([json.loads(i.strip('\n')) for i in open(t).readlines()])
    train_dataset_s = SimplifierDataset(train_gold_files,'',tokenizer,args,mode='file')
    train_dataset_r = QueryRewriteDataset(train_gold_files,'',tokenizer,args,mode='file')
    train_model_s = TrainModel(args,train_dataset_s,model_s,tokenizer,logger,contrastive_weight=0.03,cross_validate_id=i)
    train_model_r = TrainModel(args,train_dataset_r,model_r,tokenizer,logger,contrastive_weight=0.03,cross_validate_id=i)
#     train_model_r.model = pickle.load(open('train_init-2-1','rb')).model
#     with open('train_init-2-'+str(i),'wb') as f:
#         pickle.dump(train_model_r,f)
    #     -----------Phase 1-----------
    logger.info("Phase 1:")
    logger.info("Start fine-tuning simplifier with labeled data...")
    global_step, tr_loss = train_model_s.train()
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info("Start fine-tuning rewriter with labeled data...")
    global_step, tr_loss = train_model_r.train()
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
#     -----------------------------
    if i!=-1:
        output_dir_s = args.output_dir + '-initial-' + str(i) + '-s'
        output_dir_r = args.output_dir + '-initial-' + str(i) + '-r'
    else:
        output_dir_s = args.output_dir + '-initial-s'
        output_dir_r = args.output_dir + '-initial-r'
    if not os.path.exists(output_dir_s):
        os.makedirs(output_dir_s)
    if not os.path.exists(output_dir_r):
        os.makedirs(output_dir_r)
    logger.info("Saving model s checkpoint to %s", output_dir_s)
    model_to_save = train_model_s.model.module if hasattr(train_model_s.model, 'module') else train_model_s.model
    model_to_save.save_pretrained(output_dir_s)
    tokenizer.save_pretrained(output_dir_s)
    torch.save(args, os.path.join(output_dir_s, 'training_args.bin'))
    logger.info("Saving model r checkpoint to %s", output_dir_r)
    model_to_save = train_model_r.model.module if hasattr(train_model_r.model, 'module') else train_model_r.model
    model_to_save.save_pretrained(output_dir_r)
    tokenizer.save_pretrained(output_dir_r)
    torch.save(args, os.path.join(output_dir_r, 'training_args.bin'))
#     else:
#         train_dataset_s = SimplifierDataset([],'',tokenizer,args,mode='file')
#         train_dataset_r = QueryRewriteDataset([],'',tokenizer,args,mode='file')
#         train_model_s = TrainModel(args,train_dataset_s,model_s,tokenizer,logger,cross_validate_id=i)
#         train_model_r = TrainModel(args,train_dataset_r,model_r,tokenizer,logger,cross_validate_id=i)
    #     -----------Phase 2-----------
    logger.info("Phase 2:")
    labeled_data1 = copy.deepcopy(labeled_data)
    teacher_model_s = copy.deepcopy(train_model_s.model)
    teacher_model_r = copy.deepcopy(train_model_r.model)
    labeled_data_s_new = []
    labeled_data_r_new = []  
    while True:
        inference_model_s = InferenceModel(args, model=train_model_s.model,tokenizer=tokenizer,mode='model')
        inference_model_r = InferenceModel(args, model=train_model_r.model,tokenizer=tokenizer,mode='model')
        # renew
        if num!=1:
            if len(labeled_data_s_new)!=0:
                labeled_data_s_new = exchange(labeled_data_s_new)
                renew_s = predict(labeled_data_s_new ,inference_model_s,threshold=0.)
                labeled_data_s_new = exchange(labeled_data_s_new)
                start1 = labeled_data1.index(labeled_data_s_new[0])
                labeled_data1[start1:start1+len(renew_s)]=renew_s
            if len(labeled_data_r_new)!=0:
                renew_r = predict(labeled_data_r_new ,inference_model_r,threshold=0.)
                start2 = labeled_data1.index(labeled_data_r_new[0])
                labeled_data1[start2:start2+len(renew_r)]=renew_r
        print('**************')
        # generate pseudo data
#         if num>=1:
#             break
        labeled_data_s_new = predict(unlabeled_data_s ,inference_model_s,threshold=args.confidence_threshold_s-20*(num-1))#
        logger.info("Found %s valid data in S", len(labeled_data_s_new))
        labeled_data_r_new = predict(unlabeled_data_r ,inference_model_r,threshold=args.confidence_threshold_r-40*(num-1))#
        logger.info("Found %s valid data in R", len(labeled_data_r_new))
        # delete data from raw data
        unlabeled_data_s = data_removal(unlabeled_data_s, labeled_data_s_new)
        logger.info("Remain %s unlabeled data in S", len(unlabeled_data_s))
        unlabeled_data_r = data_removal(unlabeled_data_r, labeled_data_r_new)
        logger.info("Remain %s unlabeled data in R", len(unlabeled_data_r))
        labeled_data_s_new = exchange(labeled_data_s_new)
        labeled_data1.extend(labeled_data_s_new)
        labeled_data1.extend(labeled_data_r_new)
        logger.info("Having %s labeled data", len(labeled_data1))
        # Do data augmentation 
# pls fill the data_aug function and note that the augmented data does not contain labeled data
#         new_data_s = data_aug(labeled_data1, labeled_data)
        new_data_s = [i for i in labeled_data1 if i not in labeled_data]
        # Retrain the models
#         (len(labeled_data_s_new)==0 and len(labeled_data_r_new)==0) or 
        if num>=2:
            break
        train_dataset_s_1 = SimplifierDataset(labeled_data,new_data_s,tokenizer,args, mode='list')
        train_dataset_r_1 = QueryRewriteDataset(labeled_data,new_data_s,tokenizer,args, mode='list')
        train_model_s.train_dataset = train_dataset_s_1
        train_model_r.train_dataset = train_dataset_r_1
        train_model_s.contrastive_weight = args.contrastive_weight
        train_model_r.contrastive_weight = args.contrastive_weight
#         train_model_s.model = teacher_model_s
#         train_model_r.model = teacher_model_r
#         train_model_s = TrainModel(args,train_dataset_s_1,train_model_s.model,tokenizer,logger,cross_validate_id=i)
#         train_model_r = TrainModel(args,train_dataset_r_1,train_model_r.model,tokenizer,logger,cross_validate_id=i)
        global_step, tr_loss = train_model_s.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step, tr_loss = train_model_r.train()
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        logger.info("Co-training Step %s finished", str(num))
#         if num>=1:
#             break
        if i!=-1:
            output_dir_s = args.output_dir + '-' +str(i)+ '-' + str(num) + '-s'
            output_dir_r = args.output_dir + '-' +str(i)+ '-' + str(num) + '-r'
        else:
            output_dir_s = args.output_dir +'-' + str(num) + '-s'
            output_dir_r = args.output_dir +'-'+ str(num) + '-r'
        if not os.path.exists(output_dir_s):
            os.makedirs(output_dir_s)
        if not os.path.exists(output_dir_r):
            os.makedirs(output_dir_r)
        logger.info("Saving model s checkpoint to %s", output_dir_s)
        model_to_save = train_model_s.model.module if hasattr(train_model_s.model, 'module') else train_model_s.model
        model_to_save.save_pretrained(output_dir_s)
        tokenizer.save_pretrained(output_dir_s)
        torch.save(args, os.path.join(output_dir_s, 'training_args.bin'))
        logger.info("Saving model r checkpoint to %s", output_dir_r)
        model_to_save = train_model_r.model.module if hasattr(train_model_r.model, 'module') else train_model_r.model
        model_to_save.save_pretrained(output_dir_r)
        tokenizer.save_pretrained(output_dir_r)
        torch.save(args, os.path.join(output_dir_r, 'training_args.bin'))
#         if len(unlabeled_data_s)==0 or len(unlabeled_data_r)==0 :
#             break
        #         --------------------
#         output_dir_s = args.output_dir + '-' +str(i)+ '-' + str(num+1) + '-s'
#         output_dir_r = args.output_dir + '-' +str(i)+ '-' + str(num+1) + '-r'
# #         train_model_s = TrainModel(args,train_dataset_s_1,train_model_s.model,tokenizer,logger,cross_validate_id=i)
# #         train_model_r = TrainModel(args,train_dataset_r_1,train_model_r.model,tokenizer,logger,cross_validate_id=i)
#         global_step, tr_loss = train_model_s.train()
#         logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
#         global_step, tr_loss = train_model_r.train()
#         logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
#         logger.info("Saving model s checkpoint to %s", output_dir_s)
#         model_to_save = train_model_s.model.module if hasattr(train_model_s.model, 'module') else train_model_s.model
#         model_to_save.save_pretrained(output_dir_s)
#         tokenizer.save_pretrained(output_dir_s)
#         torch.save(args, os.path.join(output_dir_s, 'training_args.bin'))
#         logger.info("Saving model r checkpoint to %s", output_dir_r)
#         model_to_save = train_model_r.model.module if hasattr(train_model_r.model, 'module') else train_model_r.model
#         model_to_save.save_pretrained(output_dir_r)
#         tokenizer.save_pretrained(output_dir_r)
#         torch.save(args, os.path.join(output_dir_r, 'training_args.bin'))
#         -------------------------------
        num+=1
        # set break condition
        
    logger.info("Finished!")
    return 0
        
        
        
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name_or_path", default="gpt2-medium", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--block_size", default=150, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--train_gold_file", default=None, type=str, required=True,
                        help="Path of training file. Do not add fold suffix when cross validate, i.e. use 'data/eval_topics.jsonl' instead of 'data/eval_topics.jsonl.0'")
    parser.add_argument("--train_unlabeled_file_s", default=None, type=str, required=True,
                        help="Path of training simplifier file.")
    parser.add_argument("--train_unlabeled_file_r", default=None, type=str, required=True,
                        help="Path of training rewriter file.")
    parser.add_argument("--cross_validate", action='store_true',
                        help="Set when doing cross validation")
    parser.add_argument("--zero_shot", action='store_true',
                        help="Set when zero shot")
    parser.add_argument("--init_from_multiple_models", action='store_true',
                        help="Set when initialize from different models during cross validation (Model-based+CV)")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--confidence_threshold_s", default=60, type=float,
                        help="The initial confidence threshold for prediction Simplifier.")
    parser.add_argument("--confidence_threshold_r", default=90, type=float,
                        help="The initial confidence threshold for prediction Rewriter.")
    parser.add_argument("--aug_data_weight", default=0.5, type=float,
                        help="The loss weight of augmented weak data.")
    parser.add_argument("--contrastive_weight", default=0.03, type=float,
                        help="The contrastive learning weight.")
    parser.add_argument("--add_contrastive_loss", action='store_true',
                        help="Set when adding contrastive loss.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument("--length", type=int, default=20,
                        help="Maximum length of output sequence")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
    if args.cross_validate:
        # K-Fold Cross Validation
        for i in range(NUM_FOLD):
            logger.info("Training Fold #{}".format(i))
            suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
            config = config_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + suffix)
            tokenizer.add_special_tokens(special_tokens_dict)
            model_s = model_class.from_pretrained(args.model_name_or_path + suffix,output_hidden_states=True)
            model_s.resize_token_embeddings(len(tokenizer))  # resize
            model_s.to(args.device)
            model_r = model_class.from_pretrained(args.model_name_or_path + suffix,output_hidden_states=True)
            model_r.resize_token_embeddings(len(tokenizer))  # resize
            model_r.to(args.device)
            if args.block_size <= 0:
                args.block_size = tokenizer.max_len_single_sentence
            args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
            logger.info("Training/evaluation parameters %s", args)
            if not args.zero_shot:
                train_files = ["%s.%d" % (args.train_gold_file, j) for j in range(NUM_FOLD) if j != i]
            else:
                train_files = ['data/weak_supervision_data/rule-based.jsonl']
            train_unlabeled_s = args.train_unlabeled_file_s
            train_unlabeled_r = args.train_unlabeled_file_r
            logger.info("train_files: {}".format(train_files))
            logger.info("train_unlabeled_file_s: {}".format(train_unlabeled_s))
            logger.info("train_unlabeled_file_r: {}".format(train_unlabeled_r))
            pipeline(args,train_files,train_unlabeled_s,train_unlabeled_r,model_s, model_r, tokenizer,logger,i)
            # Create output directory if needed
            output_dir_s = args.output_dir + '-' + str(i) + '-s'
            output_dir_r = args.output_dir + '-' + str(i) + '-r'
            if not os.path.exists(output_dir_s):
                os.makedirs(output_dir_s)
            if not os.path.exists(output_dir_r):
                os.makedirs(output_dir_r)
            logger.info("Saving model s checkpoint to %s", output_dir_s)
            model_to_save = model_s.module if hasattr(model_s, 'module') else model_s
            model_to_save.save_pretrained(output_dir_s)
            tokenizer.save_pretrained(output_dir_s)
            torch.save(args, os.path.join(output_dir_s, 'training_args.bin'))
            del model_s
            logger.info("Saving model r checkpoint to %s", output_dir_r)
            model_to_save = model_r.module if hasattr(model_r, 'module') else model_r
            model_to_save.save_pretrained(output_dir_r)
            tokenizer.save_pretrained(output_dir_r)
            torch.save(args, os.path.join(output_dir_r, 'training_args.bin'))
            del model_r
            torch.cuda.empty_cache() 
#             break
    else:
        logger.info("Training: ")
        suffix = ('-' + str(i)) if args.init_from_multiple_models else ''
        config = config_class.from_pretrained(args.model_name_or_path + suffix)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path + suffix)
        tokenizer.add_special_tokens(special_tokens_dict)
        model_s = model_class.from_pretrained(args.model_name_or_path + suffix,output_hidden_states=True)
        model_s.resize_token_embeddings(len(tokenizer))  # resize
        model_s.to(args.device)
        model_r = model_class.from_pretrained(args.model_name_or_path + suffix,output_hidden_states=True)
        model_r.resize_token_embeddings(len(tokenizer))  # resize
        model_r.to(args.device)
        if args.block_size <= 0:
            args.block_size = tokenizer.max_len_single_sentence
        args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
        logger.info("Training/evaluation parameters %s", args)
        if not args.zero_shot:
            train_files = [args.train_gold_file]
        else:
            train_files = ['data/weak_supervision_data/rule-based.jsonl']
        train_unlabeled_s = args.train_unlabeled_file_s
        train_unlabeled_r = args.train_unlabeled_file_r
        logger.info("train_files: {}".format(train_files))
        logger.info("train_unlabeled_file_s: {}".format(train_unlabeled_s))
        logger.info("train_unlabeled_file_r: {}".format(train_unlabeled_r))
        pipeline(args,train_files,train_unlabeled_s,train_unlabeled_r,model_s, model_r, tokenizer,logger,-1)
        # Create output directory if needed
        output_dir_s = args.output_dir + '-s'
        output_dir_r = args.output_dir + '-r'
        if not os.path.exists(output_dir_s):
            os.makedirs(output_dir_s)
        if not os.path.exists(output_dir_r):
            os.makedirs(output_dir_r)
        logger.info("Saving model s checkpoint to %s", output_dir_s)
        model_to_save = model_s.module if hasattr(model_s, 'module') else model_s
        model_to_save.save_pretrained(output_dir_s)
        tokenizer.save_pretrained(output_dir_s)
        torch.save(args, os.path.join(output_dir_s, 'training_args.bin'))
        del model_s
        logger.info("Saving model r checkpoint to %s", output_dir_r)
        model_to_save = model_r.module if hasattr(model_r, 'module') else model_r
        model_to_save.save_pretrained(output_dir_r)
        tokenizer.save_pretrained(output_dir_r)
        torch.save(args, os.path.join(output_dir_r, 'training_args.bin'))
        del model_r
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    main()
