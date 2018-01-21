
class IdsMetric:
    def __init__(self, size, hits, tp, tn, fp, fn):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.hits = hits
        self.size = size

    def get_detection_rate(self):
        return self.tp / (self.tp + self.fn)

    def get_false_alarm_rate(self):
        return self.fn / (self.tp + self.fn)

    def get_accuracy(self):
        return self.hits / float(self.size)


def test_som_and_get_stats(som, neuron_mappings, labeled_data):
    hits = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for td in labeled_data:
        label = 'normal.' if td[-1] == 'normal.' else 'malicious.'
        winner = som.winner(td[:-1])
        pred = neuron_mappings[winner[0]][winner[1]]
        if pred == label:
            if pred == 'normal.':
                tn += 1
            else:
                tp += 1
            hits += 1
        else:
            if pred == 'normal.':
                fn += 1
            else:
                fp += 1
    return IdsMetric(len(labeled_data), hits, tp, tn, fp, fn)
