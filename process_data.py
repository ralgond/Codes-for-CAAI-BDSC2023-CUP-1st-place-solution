import pandas as pd
from ordered_set import OrderedSet
from collections import defaultdict, Counter
from sklearn.utils import shuffle
import numpy as np
import random

#===================生成entities.dict和relations.dict===================

event_df = pd.read_json("./raw_data/ecom-social/event_info.json")
rel_set = OrderedSet()
for rel in event_df['event_id']:
    rel_set.add(rel)

rel_d = {}
with open("./data/ecom-social/relations.dict", "w+") as of:
    for idx,rel in enumerate(rel_set):
        of.write(f"{idx}\t{rel}\n")
        rel_d[rel] = idx


source_train_df = pd.read_json("./raw_data/ecom-social/source_event_preliminary_train_info.json")
l = source_train_df['inviter_id'].tolist() + source_train_df['voter_id'].tolist()

target_train_df = pd.read_json("./raw_data/ecom-social/target_event_preliminary_train_info.json")
l += target_train_df['inviter_id'].tolist()
l += target_train_df['voter_id'].tolist()

target_test_df = pd.read_json("./raw_data/ecom-social/target_event_preliminary_test_info.json")
l += target_test_df['inviter_id'].tolist()

ent_set = OrderedSet(l)

ent_d = {}
with open("./data/ecom-social/entities.dict", "w+") as of:
    for idx,ent in enumerate(ent_set):
        of.write(f"{idx}\t{ent}\n")
        ent_d[ent] = idx
        
#=========================生成train0.txt和test0.txt=============================

train_df = pd.concat([source_train_df, target_train_df])
train_df[['inviter_id', 'event_id', 'voter_id']].to_csv("./data/ecom-social/train0.txt", index=False, sep='\t', header=False)

target_test_df[['triple_id', 'inviter_id', 'event_id']].to_csv("./data/ecom-social/test0.txt", index=False, sep='\t', header=False)

#========================生成top5_voter_for_every_event=======================

train_df = pd.read_csv("./data/ecom-social/train0.txt", names=['h', 'r', 't'], sep='\t')

s1 = train_df.groupby("r")[['h', 't']].apply(lambda x: x.values.tolist())

data = defaultdict(Counter)

for r, ht_l in s1.items():
    for h, t in ht_l:
        data[r].update([t])

with open("./data/ecom-social/top5_voter_for_every_event.txt", "w+") as of:
    for r, t_counter in data.items():
        l = t_counter.most_common(5)
        l2 = [str(term[0]) for term in l]
        of.write('{}\t{}\n'.format(r, "\t".join(l2)))
        
#======================将train0.txt切分成train.txt和valid.txt====================

random.seed(42)
np.random.seed(42)

data = pd.read_csv("./data/ecom-social/train0.txt", sep="\t", names=['h', 'r', 't'])

data : pd.DataFrame = shuffle(data)

size = len(data)

valid_size = int(size * 0.002)

train_df = data.iloc[:-valid_size, :]
print ("train_df.len:", len(train_df))

valid_df = data.iloc[-valid_size:, :]
print ("valid_df.len:", len(valid_df))

train_entities = set(train_df['h']) | set(train_df['t'])
train_rel = set(train_df['r'])

train_l = []
new_valid_l = []

for _, row in valid_df.iterrows():
    h, r, t = row['h'], row['r'], row['t']
    if h not in train_entities or t not in train_entities or r not in train_rel:
        train_l.append([h,r,t])
    else:
        new_valid_l.append([h,r,t])

added_train_df = pd.DataFrame(train_l, columns=['h', 'r', 't'])
new_valid_df = pd.DataFrame(new_valid_l, columns=['h', 'r', 't'])

train_df = pd.concat([train_df, added_train_df])

train_df.to_csv("./data/ecom-social/train.txt", sep="\t", index=False, header=False)
new_valid_df.to_csv("./data/ecom-social/valid.txt", sep="\t", index=False, header=False)

# ====================将test0.txt切分成test.txt, test_triple_id.txt, test2.txt================

train_df = pd.read_csv("./data/ecom-social/train0.txt", names=['h', 'r', 't'], sep='\t')

train_ent_set = set(train_df['h'].tolist() + train_df['t'].tolist())

test_df = pd.read_csv("./data/ecom-social/test0.txt", names=['triple_id', 'h', 'r'], sep='\t')

#print (train_ent_set)

with open("./data/ecom-social/test_triple_id.txt", "w+") as of_triple1, open("./data/ecom-social/test.txt", "w+") as of_test1, open("./data/ecom-social/test2.txt", "w+") as of_testnotintrain:
    for _, row in test_df.iterrows():
        triple_id, h, r = row['triple_id'], row['h'], row['r']
        if h in train_ent_set:
            of_triple1.write(f"{triple_id}\n")
            of_test1.write(f"{h}\t{r}\n")
        else:
            of_testnotintrain.write(f"{triple_id}\t{h}\t{r}\n")