lst = ['a', 'b']
with open('./info/1.txt', 'a') as f:
    for s in lst:
        f.write(s+'\n')