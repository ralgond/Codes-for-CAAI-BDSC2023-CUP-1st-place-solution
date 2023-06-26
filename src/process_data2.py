import pandas as pd
from ordered_set import OrderedSet
from collections import defaultdict, Counter
from sklearn.utils import shuffle
import numpy as np
import random

SOURCE_TRAIN_INFO_JSON = "./raw_data/ecom-social/source_event_preliminary_train_info.json"
TARGET_TRAIN_INFO_JSON_PRE = "./raw_data/ecom-social/target_event_preliminary_train_info.json"
TARGET_TRAIN_INFO_JSON_FIN = "./raw_data/ecom-social/target_event_final_train_info.json"
TARGET_TEST_INFO_JSON = "./raw_data/ecom-social/target_event_final_test_info.json"

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
    rel_d['participate'] = len(rel_d)
    of.write(f"{rel_d['participate']}\tparticipate\n")
    #rel_d['ia_ge2'] = len(rel_d)
    #of.write(f"{rel_d['ia_ge2']}\tia_ge2")

source_train_df = pd.read_json(SOURCE_TRAIN_INFO_JSON)
l = source_train_df['inviter_id'].tolist() + source_train_df['voter_id'].tolist()

target_train_df_pre = pd.read_json(TARGET_TRAIN_INFO_JSON_PRE)
l += target_train_df_pre['inviter_id'].tolist()
l += target_train_df_pre['voter_id'].tolist()

target_train_df_fin = pd.read_json(TARGET_TRAIN_INFO_JSON_FIN)
l += target_train_df_fin['inviter_id'].tolist()
l += target_train_df_fin['voter_id'].tolist()

target_test_df = pd.read_json(TARGET_TEST_INFO_JSON)
l += target_test_df['inviter_id'].tolist()

ent_set = OrderedSet(l)

ent_d = {}
with open("./data/ecom-social/entities.dict", "w+") as of:
    for idx,ent in enumerate(ent_set):
        of.write(f"{idx}\t{ent}\n")
        ent_d[ent] = idx
        
#=========================生成train0.txt和test0.txt=============================

train_df = pd.concat([source_train_df, target_train_df_pre, target_train_df_fin])
l = []
# 添加三元组(inviter_id, 'participate', voter_id)
inviter_id_idx = train_df.columns.get_loc("inviter_id")
voter_id_idx = train_df.columns.get_loc("voter_id")
#if_inviter_participate_idx = train_df.columns.get_loc("if_inviter_participate")
if_voter_participate_idx = train_df.columns.get_loc("if_voter_participate")
for _, row in train_df.iterrows():
    inviter_id, voter_id = row[inviter_id_idx], row[voter_id_idx]
    if_voter_participate = row[if_voter_participate_idx]
    if if_voter_participate:
        l.append([inviter_id, 'participate', voter_id])
participant_train_df = pd.DataFrame(l, columns=['inviter_id', 'event_id', 'voter_id'])
print ("len(participant_train_df)=======================>", len(participant_train_df))
train0 = pd.concat([participant_train_df, train_df[['inviter_id', 'event_id', 'voter_id']]])

#l2 = []
#iv_size_df = train_df.groupby(['inviter_id', 'voter_id']).size().to_frame("iv_size").reset_index()
#count = 0
#for index, row in iv_size_df.iterrows():
#    if row['iv_size'] >= 2:
#        count += 1
#        inviter_id, voter_id = row["inviter_id"], row["voter_id"]
#        l2.append([inviter_id, "ia_ge2", voter_id])
#iage2_train_df = pd.DataFrame(l2, columns=['inviter_id', 'event_id', 'voter_id'])
#print ("len(iagt2_train_df)=======================>", len(iage2_train_df))
#train0 = pd.concat([iage2_train_df, train0[['inviter_id', 'event_id', 'voter_id']]])


train0.to_csv("./data/ecom-social/train0.txt", index=False, sep='\t', header=False)

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
import os
os.environ['PYTHONHASHSEED'] = str(42)
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
