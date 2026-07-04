import random
random.seed(1314)
cols, num_ele, num_row = 200, 100, 200
for i in range(1, num_ele+1):
    nonzero = sorted(random.sample(range(cols), random.randint(3, 15)), reverse=True)
    with open(f'{i}.txt', 'w') as f:
        f.write(' '.join(map(str, nonzero)) + '\n')
for i in range(1, num_row+1):
    nonzero = sorted(random.sample(range(cols), random.randint(2, 10)), reverse=True)
    with open(f'10{i}.txt', 'w') as f:
        f.write(' '.join(map(str, nonzero)) + '\n')
print('Done')
