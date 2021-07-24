def mi(a,b,c):
    if a<0:
        if b<0: return -1
        else: return b+c
    elif b<0: return a
    elif a<(b+c): return a
    else: return (b+c)


s=[0,3,3,8,5]
v=[0,4,4,6,5]
n = len(s)
vm = 6
T=[[],[]]
for w in range(0,n*vm+1):
    if w==0: T[1].append(0)
    elif w==v[1]: T[1].append(s[1])
    else: T[1].append(-1)
print(T[1])
for i in range(2,n):
    T.append([])
    for w in range(0,n*vm+1):
        T[i].append(mi(T[i-1][w],T[i-1][w-v[i]],s[i]))
    print(T[i])
