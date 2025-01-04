import json

top_25_cwe_id = [
    "CWE-787", "CWE-79", "CWE-89", "CWE-20", "CWE-125", "CWE-78", "CWE-416", "CWE-22", "CWE-352",
    "CWE-434", "CWE-476", "CWE-502", "CWE-190", "CWE-287", "CWE-798", "CWE-862", "CWE-77", "CWE-306", "CWE-119",
    "CWE-276", "CWE-918", "CWE-362", "CWE-400", "CWE-611", "CWE-94",
]

top_25_cwe_id = top_25_cwe_id[:25]

test_set = json.load(open('../vul4c_dataset/test.json',mode='r',))
print(f'raw test set len:{len(test_set)}')

new_test_set = []
for item in test_set:
    need_add = False
    for cwe in item['cwe_list']:
        if cwe in top_25_cwe_id:
            need_add = True

    if need_add and item['vul'] == 1:
        new_test_set.append(item)


vul_cnt = 0
for item in new_test_set:
    if item['vul'] == 1:
        vul_cnt += 1

json.dump(new_test_set,open('../vul4c_dataset/test_top_cwe_25.json',mode='w'))

print(f'top25 cwe test set:{len(new_test_set)} , {vul_cnt}')