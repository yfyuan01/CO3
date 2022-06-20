from BackTranslation import BackTranslation_Baidu
import json
import copy
import time
from tqdm import tqdm
id = '20211215001029225'
pwd = 'PPH2GfHoYI17yGyMPp5u'
tmps = ['jp']
# tmps =['ru','ara','spa','de','pt','it','kor']
languages = [[] for i in range(len(tmps))]
for m,tmp in enumerate(tmps):
    languages[m] = [[] for l in range(5)]
    for i in range(2,5):
        file = open('data/eval_topics.jsonl.'+str(i)).readlines()
        file1 = open('data/eval_topics_bl.jsonl.'+tmp+'.'+str(i),'w')
        records = [json.loads(i.strip('\n')) for i in file]
        for r in tqdm(records):
            r1 = copy.deepcopy(r)
            for k,inputs in enumerate(r['input']):
                trans = BackTranslation_Baidu(appid=id, secretKey=pwd)
                result = trans.translate(inputs, src='en', tmp=tmp)
                r1['input'][k] = result.result_text
                time.sleep(1)
            trans = BackTranslation_Baidu(appid=id, secretKey=pwd)
            result = trans.translate(r['target'], src='en', tmp=tmp)
            r1["target"] = result.result_text
            time.sleep(1)
            languages[m][i].append(r1)
            file1.write(json.dumps(r1)+'\n')
        file1.close()
        print('Finished '+tmp+' '+str(i))
