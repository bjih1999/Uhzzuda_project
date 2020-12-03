from itertools import combinations
from itertools import product
from itertools import chain
sents = [[['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']],
         [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3'], ['c1', 'c2', 'c3']]]

c = list(combinations(sents, 3))
print(len(c))
list1 = []
dest_list = []
result = []
for i in range(len(c)):
    for j in range(len(c[i])):
        #print(c[i])
        item = list(product(*c[i]))
        for k in range(len(item)):
            result = list(chain(*item[k]))
            dest_list.append(result)
            result = []
temp_list = ['0']
for i in range(len(dest_list)):
    #print(i)
    for j in range(len(temp_list)):
        #print(j)
        #print(temp_list)
        if dest_list[i] == temp_list[j]:
            break;

