import torch 
import numpy as np 
import pandas as pd 

def creating_mask(length, prob):
    """
    Creating mask.

    Args:
        length: length of array to mask 
        prob:   of what proportion the array is masked out
    """
    k = int((length-1)*prob)
    mask = (torch.ones(size=(1, length)) == 0)
    if k > 0:
        rand_mat = torch.rand(1, length-1)
        k_th_quant = torch.topk(rand_mat, k, largest = False)[0][:,-1:]
        mask[:,:-1] = rand_mat <= k_th_quant
    else:
        mask[:,:-1] = torch.rand(1, length-1) <= prob
    return mask 

def mask_questions_and_contexts(questions, contexts):
    masked_batch = []
    mask_prob = np.random.choice([0.1, 0.3, 0.5])
    for q_idx, question in enumerate(questions):
        question_splits = question.split("?")[0].split(" ")
        length = len(question_splits)
        mask = creating_mask(length, mask_prob)
        
        question_splits_masked = np.array([question_splits], dtype=object)
        try:
            question_splits_masked[mask] = "<mask>"
        except:
            print(question)
            print(question_splits)
            print(mask)
            print(question_splits_masked)
                
        question_masked = ' '.join(question_splits_masked[0])
        question_masked += '?' #question mark
        question_masked_and_context = question_masked + " " + contexts[q_idx]
        
        masked_batch.append(question_masked_and_context)
    return masked_batch

def mask_questions(questions):
    """
    Mask a batch of questions

    Args:
        questions: A batch of string questions 
    
    Output:
        masked_batch: Masked version of input string questions
    """
    masked_batch = []
    mask_prob = np.random.beta(2.0, 5.0)
    for q_idx, question in enumerate(questions):
        question_splits = question.split("?")[0].split(" ")
        length = len(question_splits)
        mask = creating_mask(length, mask_prob)
        question_splits_masked = np.array([question_splits], dtype=object)
        try:
            question_splits_masked[mask] = "<mask>"
        except:
            print(question)
            print(question_splits)
            print(mask)
            print(question_splits_masked)
                
        question_masked = ' '.join(question_splits_masked[0])
        question_masked += '?' #question mark

        masked_batch.append(question_masked)
    return masked_batch

# Perturb questions (tokenized by BERT) using BART given contexts
def perturb(batch, tokenizer, tok_gen, generator, tok_para, clf, \
        args, max_seq_length, pad_on_right, num_processes = 1):
    """
    Main perturbation functionality. 

    Args:
        batch: A batch containing input_ids, attention_masks, and ground truth labels
        tokenizer: the roberta/bert tokenizer
        tok_gen: the tokenizer for the genertor
        generator: A generator that takes in masked questions, and fills perturbed questions
        tok_para: the tokenizer for the paraphrase classifier/detector
        clf: A paraphrase detector pretrained on QQP
        args: argparser dictionary
        max_seq_length: maximum sequence length for tokenizers
        pad_on_right: if the padding in tokenizer is to the right
        num_processes: for determining if the training is using multiple GPUs
    
    Returns:
        perturbed_batch: A batch of perturbed questions
        info: A list of dictionaries containing the original question, masked question, and perturbed question
        success_perturb: A boolean indicating if the perturbation is successful
        mask: A boolean indicating if the perturbation is a paraphrase
    """
    device = generator.device 
    original = tokenizer.batch_decode(batch['input_ids'])
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    contexts  = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]
    #masked_batch = mask_questions_and_contexts(questions, contexts)
    masked_batch = mask_questions(questions)
    #logger.info(f"masked batch: {masked_batch}")
    input_ids = tok_gen(masked_batch, 
                return_tensors="pt", 
                padding=True,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                truncation=True).input_ids

    

    if num_processes > 1:
        generating_func = generator.modules.generate
    else:
        generating_func = generator.generate
    
    
    perturbation = tok_gen.batch_decode(generating_func(
        input_ids.to(device), 
        num_return_sequences=1,
        no_repeat_ngram_size=3, 
        max_length=max_seq_length, 
        do_sample=True, 
        top_p = 0.95, 
        early_stopping=True
    ), skip_special_tokens=True)
    perturbation = [p.split("?")[0].replace('_', '') + '?' for p in perturbation]
    
    info = []
    for q, m, p, c in zip(questions, masked_batch, perturbation, contexts):
        info.append({
            'context': c,
            'question': q,
            'masked_q': m,
            'perturbation': p
        })
        
    success_perturb = True
    
    try:
        tokenized_new_examples = tokenizer(
            perturbation,
            contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length, 
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        ) 

        # Compute mask if the pertubation is a paraphrase 
        # in our setting, 0 : is paraphrase | 1: not paraphrase
        tokenized_pair = tok_para(
            questions,
            perturbation, 
            truncation=True,
            max_length=max_seq_length, 
            return_overflowing_tokens=False,
            return_offsets_mapping=False,
            padding="max_length" if args.pad_to_max_length else False,
        )
        
        tokenized_pair_cuda = {}
        for key in tokenized_pair:
            tokenized_pair_cuda[key] = torch.LongTensor(tokenized_pair[key]).to(device)
        clf_output = clf(**tokenized_pair_cuda)
        mask = 1 - torch.argmax(torch.softmax(clf_output.logits, axis=1), axis=1)
    except:
        tokenized_new_examples = batch 
        success_perturb =  False
        mask = torch.zeros(args.per_device_train_batch_size, dtype=torch.long).to(device)

    if success_perturb:
        sep_token_pos_pert = torch.LongTensor([input_ids.index(tokenizer.sep_token_id) for input_ids in tokenized_new_examples['input_ids']]).to(device)
        sep_token_pos_orig = torch.LongTensor([input_ids.cpu().data.numpy().tolist().index(tokenizer.sep_token_id) for input_ids in batch['input_ids']]).to(device)
        perturbed_batch = {}
        perturbed_batch['input_ids'] = torch.LongTensor(tokenized_new_examples['input_ids']).to(device).detach()
        if 'token_type_ids' in tokenized_new_examples:
            perturbed_batch['token_type_ids'] = torch.LongTensor(tokenized_new_examples['token_type_ids']).to(device).detach()
        perturbed_batch['attention_mask'] = torch.LongTensor(tokenized_new_examples['attention_mask']).to(device).detach()

        pos_diff = sep_token_pos_pert - sep_token_pos_orig
        perturbed_batch['start_positions'] = (batch['start_positions'] + pos_diff).to(device)
        perturbed_batch['end_positions'] = (batch['end_positions'] + pos_diff).to(device)

        return perturbed_batch, info, success_perturb, mask
    else:
        info = []
        for q, c in zip(questions, contexts):
            info.append({
                'context': c,
                'question': q,
                'masked_q': q,
                'perturbation': q
            })
        return batch, info, success_perturb, mask

def produce_no_answer_batch(batch, tokenizer, args, \
        max_seq_length, pad_on_right, logger, logging=False):
    device = batch['input_ids'].device
    ids = torch.range(0, args.per_device_train_batch_size-1)
    perm_ids = torch.randperm(args.per_device_train_batch_size)
    no_answer_ids = (ids != perm_ids)
    
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    original = tokenizer.batch_decode(batch['input_ids'])
    questions = [list(filter(None, x.split(sep_token)))[0].split(cls_token)[1].lstrip().rstrip() for x in original]
    contexts  = [list(filter(None, x.split(sep_token)))[1].split(sep_token)[0].lstrip().rstrip() for x in original]

    perm_contexts = np.array(contexts)[perm_ids].tolist()
    if logging:
        for q, c, gt in zip(questions, perm_contexts, no_answer_ids):
            logger.info("----Permutation Pairs----")
            logger.info(f"Question: {q}")
            logger.info(f"Context:  {c}")
            logger.info(f"NoAnswer: {gt}")

    try:
        tokenized_new_examples = tokenizer(
            questions,
            perm_contexts,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length, 
            stride=args.doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length" if args.pad_to_max_length else False,
        ) 
        batch['start_positions'][no_answer_ids] = 0
        batch['end_positions'][no_answer_ids] = 0
        batch['input_ids'] = torch.LongTensor(tokenized_new_examples['input_ids']).to(device)
        if 'token_type_ids' in tokenized_new_examples:
            batch['token_type_ids'] = torch.LongTensor(tokenized_new_examples['token_type_ids']).to(device)
        batch['attention_mask'] = torch.LongTensor(tokenized_new_examples['attention_mask']).to(device)
        return batch, no_answer_ids.to(device)
    except:
        logger.info('Failed Permutation')
        return batch, torch.zeros_like(no_answer_ids).to(device)



def batch_get_answer_tokens(start_pos, end_pos, input_ids, args):
    batch_answer_tokens = []
    for i in range(args.per_device_train_batch_size):
        if start_pos[i] > end_pos[i]:
            answer_tokens = []
        else:
            answer_tokens = input_ids[i][start_pos[i]:end_pos[i]+1].cpu().data.numpy()
        batch_answer_tokens.append(answer_tokens)
    return batch_answer_tokens

def extract_topk_answer_tokens_from_logits(tokenizer, start_logits, end_logits, input_ids, topk=10, max_answer_length=30):
    """Get topk answer tokens from start_logits and end_logits
    """
    prelim_predictions = []
    null_prediction = {
        "tokens": [tokenizer.cls_token_id],
        "score": (start_logits[0] + end_logits[0]).item(),
        "start_index": 0,
        "end_index": 0
    }
    start_indices = torch.topk(start_logits, k=topk).indices
    end_indices = torch.topk(end_logits, k=topk).indices
    for start_index in start_indices:
        for end_index in end_indices:
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index <= start_index or end_index - start_index + 1 > max_answer_length:
                continue
            tokens = input_ids[start_index:end_index+1].cpu().data.numpy()
            # remove noise from answer extraction
            if tokenizer.sep_token_id in tokens:
                continue
            prelim_predictions.append(
                {
                    "tokens": input_ids[start_index:end_index+1].cpu().data.numpy(),
                    "score": (start_logits[start_index] + end_logits[end_index]).item(),
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )
    prelim_predictions.append(null_prediction)
    predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:topk]
    
    return predictions

def batch_get_answer_tokens_topk(tokenizer, start_logits, end_logits, input_ids, args, topk=10, max_answer_length=30):
    """Get topk answer tokens from a batch of start_logits and end_logits
    """
    batch_answer_tokens = []
    for i in range(args.per_device_train_batch_size):
        batch_answer_tokens.append(extract_topk_answer_tokens_from_logits(tokenizer, start_logits[i], end_logits[i], input_ids[i], topk, max_answer_length))
    return batch_answer_tokens
    
def batch_compute_mIoU(gt_answer_tokens, p_answer_tokens_topk, args, logger, topk=10, method='mean'):
    """Compute mean IoU for a batch of ground truth answer tokens and a batch of predicted answer tokens
    This will be used as a threshold for filtering out bad perturbations
    """
    batch_mIoU = []
    for i in range(args.per_device_train_batch_size):
        answer_tokens = gt_answer_tokens[i]
        list_iou = []
        n_empty = 0
        k = min(len(p_answer_tokens_topk[i]), topk)
        if k < topk:
            logger.info(f"k: {k}")
        for j in range(k):
            p_answer_tokens =  p_answer_tokens_topk[i][j]['tokens']
            if len(p_answer_tokens) == 0: n_empty+=1 
            intersection = np.intersect1d(p_answer_tokens, answer_tokens)
            union = np.union1d(p_answer_tokens, answer_tokens)
            iou = intersection.shape[0] / union.shape[0] if union.shape[0] > 0 else 0.0
            list_iou.append(iou)
        batch_mIoU.append(list_iou)
    return batch_mIoU

def get_topk(retrieved_psgs, index, num_rows, topk = 2):
    retrieved_psgs = np.array(eval(retrieved_psgs))
    retrieved_psgs = retrieved_psgs[retrieved_psgs != index]
    retrieved_psgs = retrieved_psgs[retrieved_psgs < num_rows]
    return retrieved_psgs[:topk]

def flatten_column(df, column_name):
    repeat_lens = [len(item) if item is not np.nan else 1 for item in df[column_name]]
    df_columns = list(df.columns)
    df_columns.remove(column_name)
    expanded_df = pd.DataFrame(np.repeat(df.drop(column_name, axis=1).values, repeat_lens, axis=0), columns=df_columns)
    flat_column_values = np.hstack(df[column_name].values)
    expanded_df[column_name] = flat_column_values
    expanded_df[column_name].replace('nan', np.nan, inplace=True)
    return expanded_df