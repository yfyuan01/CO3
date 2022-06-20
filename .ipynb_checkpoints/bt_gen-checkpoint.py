import json
def deleteDuplicate(li):
    temp_list = list(set([str(i) for i in li]))
    li=[eval(i) for i in temp_list]
    return li
data = [[] for i in range(5)]
tmps =['jp','zh','fra','ru','ara','spa','de','pt','it','kor']
for i in range(5):
    file = open('data/eval_topics.jsonl.'+str(i)).readlines()
    records = [json.loads(i.strip('\n')) for i in file]
    for tmp in tmps:
        datas = open('data/eval_topics_bl.jsonl.'+tmp+'.'+str(i)).readlines()
        datas = [json.loads(k.strip('\n')) for k in datas]
        data[i].extend(datas)
    data[i] = [k for k in data[i] if '' not in k['input'] and k['target']!='']
    data[i] = deleteDuplicate([k for k in data[i] if k not in records])
    with open('data/weak_supervision_data/bt.jsonl.'+str(i),'w') as f:
        for d in data[i]:
            f.write(json.dumps(d)+'\n')
    print(len(data[i]))
    