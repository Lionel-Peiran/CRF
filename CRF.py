import numpy as np
from setting import tag2id, id2tag, unigram_template, bigram_template, mydictPath, target_epochs
from tools import gendata_fromfile, create_mydictKey, gen_ans_fromMaxScore
import pickle

class CRF:
    def __init__(self, dictPath=mydictPath):
        # scoreMap存在就先读取
        try:
            print("正在载入模型，请稍候：")
            fp = open(dictPath, "rb")
            self.crf_dict = pickle.load(fp)
            fp.close()
            print("加载模型成功")
        except IOError:
            self.crf_dict = {'Init': 1}
            print("模型不存在，初始化成功")
        self.unigram = unigram_template
        self.bigram = bigram_template

    def calUnigram(self, sentence, pos, cur_tag):
        score = 0
        for i in range(len(self.unigram)):
            key = create_mydictKey(self.unigram[i], str(i), sentence, pos, cur_tag)
            if key in self.crf_dict.keys():
                score += self.crf_dict[key]
        return score

    def calBigram(self, sentence, pos, last_tag, cur_tag):
        score = 0
        for i in range(len(self.bigram)):
            key = create_mydictKey(self.bigram[i], str(i), sentence, pos, last_tag + cur_tag)
            if key in self.crf_dict.keys():
                score += self.crf_dict[key]
        return score

    def start_train(self, epochs=target_epochs):
        sentences, tags = gendata_fromfile(FILEPATH=f"./2014_corpus.txt")
        for i in range(epochs):
            wrong_num = 0
            test = 0
            length = len(sentences)
            for j in range(len(sentences)):
                if j % 1000 == 0:
                    print(f"epoch:{i+1}/{epochs}" + f" process:{j}/{length}" + (" rate:%.2f%%" % (100 * j/length)))
                sentence = sentences[j]
                test += len(sentence)
                result = tags[j]
                wrong_num += self.train(sentence, result)
            corr_num = test - wrong_num
            self.savemydict()
            print(f'EPOCH: {i + 1} END, Dict Saved')
            print(f"epoch {i+1}" + f" accuracy: %.2f%%" % (100 * corr_num / test))

    def train(self, sentence, tags):
        predict_tags = self.getansTag(sentence)
        length = len(sentence)
        wrong_count = 0
        for i in range(length):
            if predict_tags[i] != tags[i]:
                tag_predict = predict_tags[i]
                tag_real = tags[i]
                wrong_count += 1
                for j in range(len(self.unigram)):
                    key_err = create_mydictKey(self.unigram[j], j, sentence, i, tag_predict)
                    if key_err in self.crf_dict.keys():
                        self.crf_dict[key_err] -= 1
                    else:
                        self.crf_dict[key_err] = -1
                    key_corr = create_mydictKey(self.unigram[j], j, sentence, i, tag_real)
                    if key_corr in self.crf_dict.keys():
                        self.crf_dict[key_corr] += 1
                    else:
                        self.crf_dict[key_corr] = 1
                for j in range(len(self.bigram)):
                    if 1 < i:
                        key_err = create_mydictKey(self.bigram[j], str(j), sentence, i, predict_tags[i - 1:i + 1])
                        key_corr = create_mydictKey(self.bigram[j], str(j), sentence, i, tags[i - 1:i + 1])
                    else:
                        key_err = create_mydictKey(self.bigram[j], str(j), sentence, i, tag_predict)
                        key_corr = create_mydictKey(self.bigram[j], str(j), sentence, i, tag_real)

                    if key_err in self.crf_dict.keys():
                        self.crf_dict[key_err] -= 1
                    else:
                        self.crf_dict[key_err] = -1
                    if key_corr in self.crf_dict.keys():
                        self.crf_dict[key_corr] += 1
                    else:
                        self.crf_dict[key_corr] = 1

        return wrong_count

    def getansTag(self, sentence):
        length = len(sentence)
        pre_tags = np.zeros((len(tag2id), length), dtype=str)
        max_values = np.zeros((len(tag2id), length))
        for i in range(length):
            for j in range(len(tag2id)):
                cur_tag = id2tag[j]
                if i == 0:
                    unigram_v = self.calUnigram(sentence, 0, cur_tag)
                    bigram_v = self.calBigram(sentence, 0, ' ', cur_tag)
                    max_values[j][i] = unigram_v + bigram_v
                    pre_tags[j][i] = ''
                else:
                    scores = np.zeros(len(tag2id))
                    for k in range(len(tag2id)):
                        pre_tag = id2tag[k]
                        last_word = max_values[k][i - 1]
                        unigram_v = self.calUnigram(sentence, i, cur_tag)
                        bigram_v = self.calBigram(sentence, i, pre_tag, cur_tag)
                        scores[k] = last_word + unigram_v + bigram_v
                    max_values[j][i] = np.max(scores)
                    pre_tags[j][i] = id2tag[np.argmax(scores)]
        ans = gen_ans_fromMaxScore(length, max_values, pre_tags)
        return ans


    def savemydict(self):
        with open(mydictPath, "wb") as fp:
            pickle.dump(self.crf_dict, fp)
