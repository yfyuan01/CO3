
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from cqr.utils import NUM_FOLD, set_seed, special_tokens_dict


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class InferenceModel:

    def __init__(self, args, model=None, tokenizer=None, mode='file'):
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        if mode == 'file':
            self.tokenizer = tokenizer_class.from_pretrained(args.model_path)
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model = model_class.from_pretrained(args.model_path)
        else:
            self.tokenizer = tokenizer
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model = model
        self.model.to(args.device)
        self.model.eval()

        self.device = args.device
        self.length = args.length
        if self.model.config.max_position_embeddings < args.length:
            self.length = model.config.max_position_embeddings # No generation bigger than model size 
        self.temperature = args.temperature
        self.top_p = args.top_p

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

#     def get_input_seq(self, input_sents):
#         inputs = []
#         for sent in input_sents:
#             inputs.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent)))
#             inputs.append(self.tokenizer.sep_token_id)
#         inputs.pop()
#         inputs.append(self.tokenizer.bos_token_id)
#         return inputs
    def get_input_seq(self, input_sents):
        inputs = []
        for sents in input_sents:
            inputs.append(' <SEP> '.join(sents)+' <EOS>')
        return inputs

    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text

    def predict(self, input_sents):
        self.tokenizer.padding_side = "left"
        input_sents = self.get_input_seq(input_sents)
        input_len = [len(i) for i in input_sents]
        encodings_dict = self.tokenizer.batch_encode_plus(input_sents, padding=True)
        input_ids = torch.tensor(encodings_dict['input_ids']).to(self.device)
        attn_mask = torch.tensor(encodings_dict['attention_mask']).to(self.device) 
#         input_ids = self.get_input_seq(input_sents)
#         input_length = len(input_ids)
        
#         input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
#         input_ids.to(self.device)
#         conf = 0.
        with torch.no_grad():
            outputs = self.model.generate(input_ids, attention_mask=attn_mask, top_p=self.top_p, max_length=input_ids.size(1)+self.length,return_dict_in_generate=True,output_scores=True,pad_token_id=self.tokenizer.pad_token_id)
        scores = outputs.scores
        all_score = [[] for i in range(len(input_sents))]
        for l,score in enumerate(scores):
            score = to_list(torch.max(score,dim=1)[0])
            for k,s in enumerate(score):
                all_score[k].append(s)   
        outputs = outputs.sequences
        all_sequence = [to_list(outputs[i])[input_ids.size(1):] for i in range(len(input_sents))]
        outputs = [self.tokenizer.decode(output, skip_special_tokens=False) for output in outputs]
        sequence_len = []
        for k,s in enumerate(all_sequence):
            try: 
                sequence_len.append(s.index(self.tokenizer.eos_token_id)) if s.index(self.tokenizer.eos_token_id)!=0 else sequence_len.append(20)                    
            except:
                sequence_len.append(20)
#                 print(outputs[k])
#                 print(sum(all_score[k][:sequence_len[k]+1])/(sequence_len[k]))
                
#         sequence_len = [s.index(self.tokenizer.eos_token_id) for s in all_sequence]
        all_score = [sum(all_score[i][:sequence_len[i]+1])/(sequence_len[i]+1) for i in range(len(all_score))]
        
        outputs = [output[output.find("<EOS>")+6:] for output in outputs]
        outputs = [output[:output.find("<EOS>")-1] for output in outputs]
        outputs = [self.remove_special_tokens(output) for output in outputs]
#         print(outputs)
#             for l in range(self.length):
#                 inputs = {'input_ids': input_ids}     
#                 outputs = self.model(**inputs)
#                 next_token_logits = outputs[0][:, -1, :] / (self.temperature if self.temperature > 0 else 1.)
#                 # size 1*vocab_size
#                 filtered_logits = top_p_filtering(next_token_logits, top_p=self.top_p)
#                 if self.temperature == 0: # greedy sampling:
#                     confidence, next_token = torch.max(filtered_logits, dim=-1)
# #                     confidence = confidence.unsqueeze(-1)
#                     next_token = next_token.unsqueeze(-1)
# #                     next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
# #                     confidence = torch.max(filtered_logits, dim=-1).unsqueeze(-1)
#                 else:
#                     next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
#                     confidence = (filtered_logits*F.one_hot(next_token[:,0],num_classes=filtered_logits(1))).sum(dim=1,keepdim=True)
#                 new_token = to_list(next_token)
# #                 print(to_list(confidence))
#                 confidence = sum(to_list(confidence))
#                 conf+=confidence
#                 if self.tokenizer.decode(new_token[0]).strip() == "<EOS>":
#                     break
#                 input_ids = torch.cat((input_ids, next_token), dim=1)
#         conf = conf/(l+1)
#         pred_ids = to_list(input_ids[0, input_length:])
#         pred_text = self.tokenizer.decode(pred_ids, clean_up_tokenization_spaces=True)
#         pred_text = self.remove_special_tokens(pred_text)
#         return pred_text,conf
        return outputs,all_score

