import pandas as pd

# 将predict1.txt、triple1.txt、test1.txt、test_notintrain.txt、top5_voter_for_every_event.txt全部合并回来

# 读取triple1.txt 读取predict1.txt 然后合并
triple_df = pd.read_csv("./data/ecom-social/test_triple_id.txt", names=['triple_id'])

save_path = "save/ecom-social"

candidate_df = None
with open(f"./{save_path}/predict.txt") as fin:
    data = []
    for line in fin:
        line2 = []
        term = line.strip().split()
        for i in range(0, len(term), 2):
            voter_id = term[i][:-1]
            line2.append(voter_id)
        data.append(','.join(line2))
    candidate_df = pd.DataFrame(data, columns=['candidates'])

candidate_df = pd.concat([triple_df, candidate_df], axis=1)

# 读取test_notintrain.txt、top5_voter_for_every_event.txt

top5_voter_for_every_event_d = {}
with open ("./data/ecom-social/top5_voter_for_every_event.txt") as fin:
    for line in fin:
        row = line.split()
        top5_voter_for_every_event_d[row[0]] = ','.join(row[1:])

test_notintrain_df = pd.read_csv("./data/ecom-social/test2.txt", sep='\t', names=['triple_id', 'h', 'r'])
test_notintrain_candidate_l = []
for _, row in test_notintrain_df.iterrows():
    triple_id, h, r = row['triple_id'], row['h'], row['r']
    top5_voter_for_every_event = top5_voter_for_every_event_d[r]
    test_notintrain_candidate_l.append(top5_voter_for_every_event)
test_notintrain_df['candidates'] = test_notintrain_candidate_l

del test_notintrain_df['h']
del test_notintrain_df['r']

all_df = pd.concat([candidate_df, test_notintrain_df])
all_df.sort_values('triple_id', inplace=True)

# ===========================================================
result_l = []
for _, row in all_df.iterrows():
    triple_id, candidates = row['triple_id'], row['candidates']
    d = {}
    d['triple_id'] = "%04d" % triple_id
    d['candidate_voter_list'] = candidates.split(',')

    result_l.append(d)

import json
with open("./{save_path}/submit.json", "w+") as of:
    json.dump(result_l, of)
