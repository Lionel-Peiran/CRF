from CRF import CRF
import sys
from setting import mydictPath
from tools import printwords

def Usage():
    print("Usage : ")
    print("\'python3 main.py --train\' 开始训练CRF模型，配置在setting.py")
    print("\'python3 main.py --play\' 开始交互，输入句子获取分词序列")

if __name__ == "__main__":
    if len(sys.argv) <= 1 or len(sys.argv) > 2:
        Usage()
        exit()
    train = False
    if sys.argv[1] == "--train":
        train = True
    elif sys.argv[1] == "--play":
        try:
            f = open(mydictPath, "rb")
            f.close()
        except:
            print("模型不存在，先使用train命令进行训练：")
            Usage()
            exit()
    else:
        Usage()
        exit()
    CRF = CRF()
    if train:
        print("开始训练：")
        CRF.start_train()
    else:
        print("请开始输入句子，无输入回车表示结束：")
        instr = input()
        while instr:
            tags = CRF.getansTag(instr)
            print(tags)
            printwords(instr, tags)
            print("")
            instr = input()

