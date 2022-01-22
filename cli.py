from classifier import Classifier
import time

def main():
    clf = Classifier("emsi_nb_multi.sav")
    
    while (1):
        s = input("Enter posting: ")
        if (s == 'exit'):
            break

        s = s.replace("\\n", "\n")
        segments = clf.segment(s)
        print("\n")
        for key, value in segments.items():
            print(key, ": ", value)
            input()
        print('-----------')

if __name__ == '__main__':
    main()