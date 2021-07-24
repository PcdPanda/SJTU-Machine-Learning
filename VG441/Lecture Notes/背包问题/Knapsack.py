class item:
    name = ''
    value = 0 
    size = 0

    def __init__(self, name, v, s):
        self.name = name
        self.value = v
        self.size = s
# value, size
a = item('a', 5, 5)
b = item('b', 20, 10)
c = item('c', 10, 20)
d = item('d', 30, 25)
B = 30
# 生成权重列
k = [a, b, c, d]
k.sort(key = lambda item: item.value/item.size, reverse=True)
print(k[0].name)
# 贪婪算法
print('start greedy algorithm')
total_size = 0
total_value = 0
Greedy = []
for i in range(0, len(k)):
    if total_size + k[i].size <= B:
        Greedy.append(k[i])
        total_size += k[i].size
        total_value += k[i].value
for i in range(0, len(Greedy)):
    print(Greedy[i].name)
print('Total size=', total_size, 'total value=', total_value)

# 2阶约化算法
print('start 2-approx algorithm')
total_size = 0
total_value = 0
Greedy = []
for i in range(0, len(k)):
    if total_size + k[i].size <= B:
        Greedy.append(k[i])
        total_size += k[i].size
        total_value += k[i].value
if i != len(k):
    if k[i].value > total_value:
        total_size = k[i].size
        total_value = k[i].value
        print(k[i].name)
    else :
        for i in range(0, len(Greedy)):
            print(Greedy[i].name)
print('Total size=', total_size, 'total value=', total_value)