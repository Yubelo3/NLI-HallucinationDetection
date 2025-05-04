def accuracy(sample_count, true_pred, false_pred):
    return sum(true_pred)/sum(sample_count)


def precision(sample_count, true_pred, false_pred):
    return [x/(x+y) for x, y in zip(true_pred, false_pred)]


def recall(sample_count, true_pred, false_pred):
    return [x/y for x, y in zip(true_pred, sample_count)]


def f1_score(sample_count, true_pred, false_pred):
    p = precision(sample_count, true_pred, false_pred)
    r = recall(sample_count, true_pred, false_pred)
    return [2*x*y/(x+y) for x, y in zip(p, r)]
