import sys
import os

path = sys.argv[1]

fn_list = os.listdir(path)

l = []

for fn in fn_list:
    if fn.startswith("predict_"):
        terms = fn.split('.txt')[0].split("_")
        mrr = float(terms[2])
        l.append((fn, mrr))

l2 = sorted(l, key=lambda x: x[1], reverse=True)

print (l2)
