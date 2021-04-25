import re
import random
import numpy as np
from setting import DataFilePath, Scale, tag2id, id2tag
rule = r'/\w+|[\[\]]'

def printwords(instr,tags):
    for i in range(len(instr)):
        print(instr[i],end='')
        if tags[i] == 'E' or tags[i] == 'S':
            print(' ',end='')
    print('')


def gen_ans_fromMaxScore(length, max_score, pre_tags):
    res_buf = [0] * length
    score_buf = np.zeros(len(tag2id))
    for i in range(len(tag2id)):
        score_buf[i] = max_score[i][length - 1]
    res_buf[length - 1] = id2tag[np.argmax(score_buf)]
    for back_idx in range(length - 2, -1, -1):
        res_buf[back_idx] = pre_tags[tag2id[res_buf[back_idx + 1]]][back_idx + 1]
    ans = ''
    for i in range(length):
        ans = ans + res_buf[i]
    return ans


def word2tag(tagstr):
    tags = []
    if len(tagstr) == 1:
        tags.append('S')
    elif len(tagstr) == 2:
        tags = ['B', 'E']
    else:
        M_num = len(tagstr) - 2
        M_list = ['M'] * M_num
        tags.append('B')
        tags.extend(M_list)
        tags.append('E')
    return "".join(tags)

def gendata_fromfile(FILEPATH=DataFilePath, scale=Scale):
    with open(f"./{FILEPATH}", "r", encoding="utf-8") as reader:
        aline = reader.readline()
        sentence = []
        tags = []
        f = open(f"./Curdata.txt", "w", encoding="utf-8")
        while aline:
            if random.randint(0, 100) in range(Scale):
                aline = re.sub(rule, "", aline)
                linelist = aline.split(" ")
                linelist.pop()
                temptag = ''
                for i in linelist:
                    temptag = temptag + word2tag(i)
                instr = "".join(linelist)
                sentence.append(instr)
                tags.append(temptag)
                f.write(instr + '\n')
                f.write(aline)
            aline = reader.readline()
        f.close()
        print("sentence num : " + str(len(sentence)))
        return sentence, tags


def create_mydictKey(template, id, sentence, pos, status_covered):
    key = str(id)
    for i in template:
        index = pos + i
        if index < 0 or index > len(sentence) - 1:
            key += ' '
        else:
            key += sentence[index]
    key += '/' + status_covered
    return key