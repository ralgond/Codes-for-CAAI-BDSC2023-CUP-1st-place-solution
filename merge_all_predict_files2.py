import sys

path = sys.argv[1]
fn1 = sys.argv[2]
fn2 = sys.argv[3]
fn3 = sys.argv[4]

print (path, fn1, fn2, fn3)

with open(f"{path}/{fn1}") as inf1, open(f"{path}/{fn2}") as inf2, \
     open(f"{path}/{fn3}") as inf3, open(f"{path}/predict.txt", "w+") as of:
    lines1 = inf1.readlines()
    lines2 = inf2.readlines()
    lines3 = inf3.readlines()
    
    for line1, line2, line3 in zip(lines1, lines2, lines3):
        d = {}
        terms = line1.strip().split('\t')
        terms.extend(line2.strip().split('\t'))
        terms.extend(line3.strip().split('\t'))
        for term in terms:
            k,v = term.split(': ')
            v = float(v)
            if k in d:
                if d[k] > v:
                    d[k] = v
            else:
                d[k] = v
                
        l = [(k,v) for k,v in d.items()]
        l = sorted(l, key=lambda x: x[1])
        s = '\t'.join(["{}: {}".format(k,v) for k,v in l[:5]])
        of.write(s+"\n")
        
