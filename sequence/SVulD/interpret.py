import torch
from transformers import RobertaTokenizer
from argparse import Namespace

# def attention_interpretation(model,tokenizer:RobertaTokenizer,mini_batch , args:Namespace):
def attention_interpretation(input_ids,attentions,tokenizer:RobertaTokenizer ):
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    raw_func = tokenizer.convert_tokens_to_string(all_tokens,)
    raw_func_loc = len(raw_func.split('\n'))
    # all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    # all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    # tab = ĉ
    # \n= Ċ
    # print(tokenizer.decode(input_ids))
    # print(all_tokens)
    attention = None
    for attention_layer in attentions:
        sum_attention = torch.sum(attention_layer,0)
        if attention is None:
            attention = sum_attention
        else:
            attention += sum_attention
    attention = clean_special_token_values(input_ids , attention , tokenizer )
    # print(len(all_tokens))
    # print(all_tokens)
    token_attn_score = [  (x[0],x[1].item()) for x in zip(all_tokens, attention) ]
    # print(token_attn_score)


    # get word level attention scores
    # separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    separator = ["Ċ", "ĊĊ"]
    special_token = ['<s>' , '</s>' , '<pad>' ,'<unk>']
    tab_token = 'ĉ'
    word_level_scores = []
    cur_word_score = 0.0
    cur_word = ""

    for idx  ,(word , score) in enumerate(token_attn_score) :
        if (word in special_token) or (word in separator) or (word == tab_token):
            if len(cur_word) !=0 :
                word_level_scores.append((cur_word, cur_word_score))
                cur_word = ""
                cur_word_score = 0.0

            word_level_scores.append( ( word, score))
        elif word[0] == 'Ġ':
            word_level_scores.append( ( cur_word, cur_word_score))
            cur_word = ""
            cur_word_score = 0.0

            cur_word += word[1:]
            cur_word_score += score
        else:
            cur_word += word
            cur_word_score += score


    # print('***' * 20)
    # print(word_level_scores)

    # get line attention scores
    line_scores = []
    cur_line_score = 0.0
    cur_line = ""
    for word , score in word_level_scores :
        if word in special_token:
            if len(cur_line) != 0:
                line_scores.append((cur_line, cur_line_score))
                cur_line = ""
                cur_line_score = 0.0
        elif word in separator:
            if len(cur_line) != 0:
                cur_line += '\n'
                line_scores.append((cur_line,cur_line_score))
                cur_line = ""
                cur_line_score = 0.0
                if len(word) > 1:  # multi line break e.g. `\n\n`
                    for _ in range(len(separator) - 1) :
                        line_scores.append(("\n",0))
            else:
                for _ in range(len(word)):
                    line_scores.append(("\n", 0))
        else:
            word = '\r' if word == tab_token else word
            cur_line += word if len(cur_line) == 0 else f" {word}"
            cur_line_score += score

    # print(''.join([x[0] for x in line_scores]))
    # print(line_scores)


    if len(line_scores) != raw_func_loc:
        # add line break in the end
        assert len(line_scores) < raw_func_loc , print( f'line_scores:{len(line_scores)} < raw_func_loc{raw_func_loc}')
        maybe_line_break = raw_func.split('\n')[-( raw_func_loc - len(line_scores)   ) :]
        for line in maybe_line_break:
            for token in special_token: # remove <s> </s>
                line = line.replace(token,"")
            if len(line.strip()) == 0:
                line_scores.append((line, 0.0))


    len_a = raw_func_loc
    len_b = len(line_scores)
    assert len_a == len_b , print(f'{len_a} != {len_b} ')

    line_token_level_scores = []
    single_line_token_level_scores = []
    for word , score in word_level_scores :
        if word in special_token or word in separator:
            if len(single_line_token_level_scores) != 0:
                line_token_level_scores.append(single_line_token_level_scores)
                single_line_token_level_scores = []
            continue
        else:
            word = '\r' if word == tab_token else word
            single_line_token_level_scores.append((word,score))

    return line_scores , line_token_level_scores

    # get line attention scores
    # line_scores = []
    # cur_line_score = 0.0
    # cur_line = ""
    #
    # for idx  ,(token , score) in enumerate(token_attn_score) :
    #     if token in special_token:
    #         continue
    #
    #     if token in separator:
    #         line_scores.append((cur_line,cur_line_score))
    #         cur_line = ""
    #         cur_line_score = 0.0
    #     else:
    #         cur_line += token
    #         cur_line_score += score
    #
    #     if idx == len(token_attn_score) - 1:
    #         line_scores.append((cur_line, cur_line_score))
    #
    # print(line_scores)

# ĊĊ  Ċ


def clean_special_token_values(input_ids :list[int], attention, tokenizer : RobertaTokenizer):
    # special token in the beginning of the seq
    special_tokens = [tokenizer.bos_token_id , tokenizer.eos_token_id , tokenizer.unk_token_id , tokenizer.pad_token_id]
    for idx,token_id in enumerate(input_ids):
        if token_id in special_tokens:
            attention[idx] = 0
    return attention

# print(RobertaTokenizer.from_pretrained('microsoft/codebert-base').tokenize('\r\r'))




