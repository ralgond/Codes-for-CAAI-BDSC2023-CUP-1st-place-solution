import sys


with open(f"{path}/predict1.txt") as inf1, open(f"{path}/predict2.txt") as inf2, \
     open(f"{path}/predict3.txt") as inf3, open(f"{path}/predict4.txt") as inf4, open(f"{path}/predict.txt", "w+") as of:
    lines1 = inf1.readlines()
    lines2 = inf2.readlines()
    lines3 = inf3.readlines()
    lines4 = inf4.readlines()
    
    for line1, line2, line3, line4 in zip(lines1, lines2, lines3, lines4):
        d = {}
        terms = line1.strip().split('\t')
        terms.extend(line2.strip().split('\t'))
        terms.extend(line3.strip().split('\t'))
        terms.extend(line4.strip().split('\t'))
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
        
