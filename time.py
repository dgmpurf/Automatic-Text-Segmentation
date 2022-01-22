from classifier import Classifier
import time

def main():
    clf = Classifier("emsi_nb_multi.sav")

    s = input("Enter posting: ")
    s = s.replace("\\n", "\n")

    start = time.time()
    docs = 0
    while (docs < 1000):
        segments = clf.segment(s)
        docs += 1

    end = time.time()
    print(docs / (end - start))

if __name__ == '__main__':
    main()