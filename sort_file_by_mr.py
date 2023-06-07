import sys
import os

path = sys.argv[1]

fn_list = os.listdir(path)

l = []

for fn in fn_list:
    if fn.startswith("predict_"):
        terms = fn.split("_")
        mr = float(terms[1])
        l.append((fn, mr))

l2 = sorted(l, key=lambda x: x[1])

print (l2)