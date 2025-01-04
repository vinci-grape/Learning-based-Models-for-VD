import json
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score, \
    roc_auc_score, precision_recall_curve, auc , average_precision_score


# prompt_result = json.load(open('./prompt_result_0.json'))
# test_result = json.load(open('../vul4c_dataset/test.json'))
#
# assert len(prompt_result) == len(test_result)
#
# labels = [func['vul'] for func in test_result]
# preds = []
# for p in prompt_result:
#     out: str = p['output']
#     out = out.replace('.', '').replace(',', '').replace('`', '').lower().strip()
#     pred = 0
#     if ('yes' in out) or ('is vulnerable' in out) or ('is a vulnerability' in out) or ('be vulnerable' in out) or (
#             'be a vulnerable' in out) \
#             or ('is a vulnerable' in out) or ('there is a potential vulnerability' in out) or (
#             'has a vulnerability' in out) \
#             or ('a few potential vulnerabilities' in out) \
#             :
#         pred = 1
#     else:
#         pred = 0
#     preds.append(pred)
#
# metrics = {}
# metrics["acc"] = accuracy_score(labels, preds)
# metrics["f1"] = f1_score(labels, preds, zero_division=0)
# metrics["rec"] = recall_score(labels, preds, zero_division=0)
# metrics["prec"] = precision_score(labels, preds, zero_division=0)
# metrics["roc auc"] = roc_auc_score(labels, preds)
#
# print(metrics)


def result_classifier(filename):
    prompt_result: list[dict] = json.load(open(filename, mode='r'))
    labels = [func['vul'] for func in prompt_result]

    pred_cwe_hit = 0
    cwe_output_detect = False
    preds = []
    cwe_preds = []
    cwe_list_labels = []
    for p in prompt_result:
        out: str = p['output']
        pred = 0
        pred_cwe = ""

        if out.find('{') != -1 and out.find('}') != -1 and out.find('yes') == -1:
            out = out[out.find('{') : out.find('}') + 1]

        if check_is_json(out):
            cwe_output_detect = True
            out = json.loads(out)
            out['vulnerable'] = out['vulnerable'].lower()
            pred = 1 if out['vulnerable'] == 'yes' else 0
            if pred == 1 and 'cwe' in out:
                pred_cwe = out['cwe'].lower()
            else:
                pred = 0

        else:
            out = out.replace('.', '').replace(',', '').replace('`', '').lower().strip()
            if ('yes' in out) or ('is vulnerable' in out) or ('is a vulnerability' in out) or ('be vulnerable' in out) or (
                    'be a vulnerable' in out) \
                    or ('is a vulnerable' in out) or ('there is a potential vulnerability' in out) or (
                    'has a vulnerability' in out) \
                    or ('a few potential vulnerabilities' in out) \
                    :
                pred = 1
            else:
                pred = 0

        preds.append(pred)
        cwe_preds.append(pred_cwe)
        cwe_list_labels.append(p['cwe_list'])
    metrics = {}
    metrics["acc"] = round(accuracy_score(labels, preds), 4)
    metrics["rec"] = round(recall_score(labels, preds, zero_division=0), 4)
    metrics["prec"] = round(precision_score(labels, preds, zero_division=0), 4)
    metrics["f1"] = round(f1_score(labels, preds, zero_division=0), 4)
    try:
        metrics["roc auc"] = round(roc_auc_score(labels, preds), 4)
    except ValueError:
        pass
    metrics["pr auc"] = round(average_precision_score(labels, preds), 4)


    if cwe_output_detect:
        cwe_acc,top10_success_cwe,top10_fail_cwe = cwe_acc_score(preds,labels,cwe_preds, cwe_list_labels)

        metrics['cwe_acc'] = round(cwe_acc,4)
        metrics['cwe_vul_acc'] = round(cwe_vul_acc_score(preds,labels,cwe_preds, cwe_list_labels),4)
        metrics['top10_success_cwe'] = top10_success_cwe
        metrics['top10_fail_cwe'] = top10_fail_cwe

    return metrics


def cwe_acc_score( preds:list,labels:list, cwe_preds:list,cwe_list_labels:list):
    hit_cnt = 0
    cwe_success_dict = {}
    cwe_fail_dict = {}
    for i in range(len(labels)):
        pred = preds[i]
        label = labels[i]
        cwe_list_label = cwe_list_labels[i]
        if pred == label:
            if pred == 1:
                cwe_pred : str= cwe_preds[i]
                cwe_found = False
                for cwe in cwe_list_label:
                    if cwe.lower() == cwe_pred:
                        hit_cnt += 1
                        cwe_found = True
                        break

                if cwe_found:
                    cwe_success_dict.setdefault(cwe_pred.upper(), 0)
                    cwe_success_dict[cwe_pred.upper()] += 1
                else:
                    for cwe in cwe_list_label:
                        cwe_fail_dict.setdefault(cwe,0)
                        cwe_fail_dict[cwe] += 1
            else:
                hit_cnt += 1
        else:
            if label == 1:
                for cwe in cwe_list_label:
                    cwe_fail_dict.setdefault(cwe, 0)
                    cwe_fail_dict[cwe] += 1

    cwe_success_dict = sort_dict_by_value_top10(cwe_success_dict)
    cwe_fail_dict = sort_dict_by_value_top10(cwe_fail_dict)


    return hit_cnt / len(preds) , cwe_success_dict , cwe_fail_dict

def sort_dict_by_value_top10(dict):
    value_key_pairs = ((value, key) for (key, value) in dict.items())
    sorted_value_key_pairs = sorted(value_key_pairs, reverse=True)[:10]
    return {k: v for v, k in sorted_value_key_pairs}

def cwe_vul_acc_score( preds:list,labels:list, cwe_preds:list,cwe_list_labels:list):
    hit_cnt = 0
    for i in range(len(labels)):
        pred = preds[i]
        label = labels[i]
        if pred == label:
            if pred == 1:
                cwe_pred = cwe_preds[i]
                cwe_list_label = cwe_list_labels[i]
                for cwe in cwe_list_label:
                    if cwe.lower() == cwe_pred:
                        hit_cnt += 1
                        break

    return hit_cnt / sum(preds) if sum(preds) != 0 else 0


def check_is_json(s: str):
    try:
        json.loads(s)
    except json.decoder.JSONDecodeError:
        return False
    return True


print(result_classifier('result/chatgpt/icl/same_repo/raw_0.0.json'))