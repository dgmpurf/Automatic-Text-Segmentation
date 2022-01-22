from classifier import Classifier

# utility function for parsing data files
def get_data(filename):
    labels = []
    segments = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            split = line.split("\t")
            labels.append(split[0])
            segments.append(split[1])

    return segments, labels

def main():
    segments, labels = get_data("data/base.txt")
    clf = Classifier()
    clf.train(segments, labels)
    clf.scores(plot_matrix=True)
    clf.save("emsi_nb_new.sav")

if __name__ == '__main__':
    main()