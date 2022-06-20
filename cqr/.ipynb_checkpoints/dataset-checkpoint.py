import copy
import json

from torch.utils.data import Dataset

class ConvSearchExample:
    def __init__(self, topic_number, query_number, ids, labels, pred_begin_pos, data_type,target):
        self.topic_number = topic_number
        self.query_number = query_number
        self.ids = ids
        self.labels = labels
        self.pred_begin_pos = pred_begin_pos
        self.data_type = data_type
        self.target = target
    
    def __repr__(self):
        print('===ConvSearchExample===')
        print(self.topic_number + '_' + self.query_number)
        print('-----------------------')
        print(self.ids)
        print('-----------------------')
        print(self.labels)
        print('-----------------------')
        print(self.pred_begin_pos)
        print('-----------------------')
        print(self.data_type)
        print('-----------------------')
        print(self.target)
        print('=======================')
        return ''


class QueryRewriteDataset(Dataset):
    def __init__(self, gold_filenames, weak_filenames, tokenizer, args, mode='file'):
        self.examples = []
        if mode == 'file':
            filenames = copy.deepcopy(gold_filenames)
            if weak_filenames!='':
                filenames.extend(weak_filenames)
            for filename in filenames:
                with open(filename, encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        input_sents = record['input']
                        target_sent = record['target']
                        topic_number = record['topic_number']
                        query_number = record['query_number']
                        data_type = 1 if filename in gold_filenames else 0
                        this_example = []
                        this_example_labels = []
                        this_example_target = []
                        for sent in input_sents:
                            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                            this_example.append(tokenizer.sep_token_id)
                        this_example.pop()
                        this_example.append(tokenizer.bos_token_id)

                        begin_pos = len(this_example)
                        this_example_labels.extend([-100] * begin_pos)
                        this_example_target.extend([tokenizer.pad_token_id] * begin_pos)
                        this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example_target.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example.append(tokenizer.eos_token_id)
                        this_example_labels.append(tokenizer.eos_token_id)
                        this_example_target.append(tokenizer.eos_token_id)
                        if len(this_example) > args.block_size:
                            this_example = this_example[:args.block_size]
                            this_example_labels = this_example_labels[:args.block_size]
                            this_example_target = this_example_target[:args.block_size]
                        else:
                            pad_num = args.block_size - len(this_example)
                            this_example.extend([tokenizer.pad_token_id] * pad_num)
                            this_example_labels.extend([-100] * pad_num)
                            this_example_target.extend([tokenizer.pad_token_id] * pad_num)
                        assert len(this_example) == args.block_size
                        assert len(this_example_labels) == args.block_size
                        self.examples.append(ConvSearchExample(topic_number, query_number, this_example, this_example_labels, begin_pos,data_type,this_example_target))
        elif mode == 'list':
            gold_list = gold_filenames
            weak_list = weak_filenames
            all_list = copy.deepcopy(gold_list)
            all_list.extend(weak_list)
            for record in all_list:
                input_sents = record['input']
                target_sent = record['target']
                topic_number = record['topic_number']
                query_number = record['query_number']
                data_type = 1 if record in gold_list else 0
                this_example = []
                this_example_labels = []
                this_example_target = []
                for sent in input_sents:
                    this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                    this_example.append(tokenizer.sep_token_id)
                this_example.pop()
                this_example.append(tokenizer.bos_token_id)
                begin_pos = len(this_example)
                this_example_labels.extend([-100] * begin_pos)
                this_example_target.extend([tokenizer.pad_token_id] * begin_pos)
                this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example_target.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example.append(tokenizer.eos_token_id)
                this_example_labels.append(tokenizer.eos_token_id)
                this_example_target.append(tokenizer.eos_token_id)
                if len(this_example) > args.block_size:
                    this_example = this_example[:args.block_size]
                    this_example_labels = this_example_labels[:args.block_size]
                    this_example_target = this_example_target[:args.block_size]
                else:
                    pad_num = args.block_size - len(this_example)
                    this_example.extend([tokenizer.pad_token_id] * pad_num)
                    this_example_labels.extend([-100] * pad_num)
                    this_example_target.extend([tokenizer.pad_token_id] * pad_num)
                assert len(this_example) == args.block_size
                assert len(this_example_labels) == args.block_size
                self.examples.append(ConvSearchExample(topic_number, query_number, this_example, this_example_labels, begin_pos,data_type,this_example_target))
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

class SimplifierDataset(Dataset):
    def __init__(self, gold_filenames, weak_filenames, tokenizer, args, mode='file'):
        self.examples = []
        if mode == 'file':
            filenames = copy.deepcopy(gold_filenames)
            if weak_filenames!='':
                filenames.extend(weak_filenames)
            for filename in filenames:
                with open(filename, encoding="utf-8") as f:
                    for line in f:
                        record = json.loads(line)
                        input_sents = copy.deepcopy(record['input'])
                        input_sents[-1] = record['target']
                        target_sent = record['input'][-1]
                        topic_number = record['topic_number']
                        query_number = record['query_number']
                        data_type = 1 if filename in gold_filenames else 0
                        this_example = []
                        this_example_labels = []
                        this_example_target = []
                        for sent in input_sents:
                            this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                            this_example.append(tokenizer.sep_token_id)
                        this_example.pop()
                        this_example.append(tokenizer.bos_token_id)

                        begin_pos = len(this_example)
                        this_example_labels.extend([-100] * begin_pos)
                        this_example_target.extend([tokenizer.pad_token_id] * begin_pos)
                        this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example_target.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                        this_example.append(tokenizer.eos_token_id)
                        this_example_labels.append(tokenizer.eos_token_id)
                        this_example_target.append(tokenizer.eos_token_id)
                        if len(this_example) > args.block_size:
                            this_example = this_example[:args.block_size]
                            this_example_labels = this_example_labels[:args.block_size]
                            this_example_target = this_example_target[:args.block_size]
                        else:
                            pad_num = args.block_size - len(this_example)
                            this_example.extend([tokenizer.pad_token_id] * pad_num)
                            this_example_labels.extend([-100] * pad_num)
                            this_example_target.extend([tokenizer.pad_token_id] * pad_num)
                        assert len(this_example) == args.block_size
                        assert len(this_example_labels) == args.block_size
                        self.examples.append(ConvSearchExample(topic_number, query_number, this_example, this_example_labels, begin_pos,data_type,this_example_target))
        elif mode == 'list':
            gold_list = gold_filenames
            weak_list = weak_filenames
            all_list = copy.deepcopy(gold_list)
            all_list.extend(weak_list)
            for record in all_list:
                input_sents = copy.deepcopy(record['input'])
                input_sents[-1] = record['target']
                target_sent = record['input'][-1]
                topic_number = record['topic_number']
                query_number = record['query_number']
                data_type = 1 if record in gold_list else 0
                this_example = []
                this_example_labels = []
                this_example_target = []
                for sent in input_sents:
                    this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)))
                    this_example.append(tokenizer.sep_token_id)
                this_example.pop()
                this_example.append(tokenizer.bos_token_id)
                begin_pos = len(this_example)
                this_example_labels.extend([-100] * begin_pos)
                this_example_target.extend([tokenizer.pad_token_id] * begin_pos)
                this_example.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example_labels.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example_target.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target_sent)))
                this_example.append(tokenizer.eos_token_id)
                this_example_labels.append(tokenizer.eos_token_id)
                this_example_target.append(tokenizer.eos_token_id)
                if len(this_example) > args.block_size:
                    this_example = this_example[:args.block_size]
                    this_example_labels = this_example_labels[:args.block_size]
                    this_example_target = this_example_target[:args.block_size]
                else:
                    pad_num = args.block_size - len(this_example)
                    this_example.extend([tokenizer.pad_token_id] * pad_num)
                    this_example_labels.extend([-100] * pad_num)
                    this_example_target.extend([tokenizer.pad_token_id] * pad_num)
                assert len(this_example) == args.block_size
                assert len(this_example_labels) == args.block_size
                self.examples.append(ConvSearchExample(topic_number, query_number, this_example, this_example_labels, begin_pos,data_type,this_example_target))
            

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]