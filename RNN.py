import collections
import re
from d2l import torch as d2l

d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()

    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
'''
对每一行调用正则替换：把所有非英文字母的字符都替换成空格。
用 strip() 去掉首尾空白。
用 lower() 统一转成小写。
最外层的列表推导式会对 lines 里的每一行都执行这个处理，最后返回一个清洗后的字符串列表。
'''

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Invalid type for token:' + token)

lines = read_time_machine()
tokens = tokenize(lines)
for i in range(20):
    print(tokens[i])