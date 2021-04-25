DataFilePath = "./2014_corpus.txt"
Scale = 2

tag2id = {'S': 0, 'B': 1, 'M': 2, 'E': 3}
id2tag = {0: 'S', 1: 'B', 2: 'M', 3: 'E'}

target_epochs = 1

unigram_template = [[-1], [0], [1], [-1, 0], [0, 1]]

bigram_template = [[-1], [0], [1], [-1, 0], [0, 1]]

mydictPath = f"./MyDict.pkl"
