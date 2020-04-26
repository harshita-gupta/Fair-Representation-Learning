import pandas as pd
import numpy as np
import csv
from sklearn.metrics import roc_curve, auc
import random
import matplotlib.pyplot as plt
import pylab as plb
import pdb


# import cPickle


# split data
# df -- dataframe
# split -- ratio to split out float (0, 1)
# seed -- random seed to use
def split_data(df, split, seed=1):
    random.seed(seed)
    rows = random.sample(df.index, int(round(split * (df.shape[0]))))
    df_split = df.ix[rows]
    df_remaining = df.drop(rows)
    return df_split, df_remaining, rows


def hump_variable(df, variable, split_pt):
    inds1 = df[variable] <= split_pt
    inds2 = df[variable] > split_pt
    df[variable + '_1'] = np.zeros(df.shape[0])
    df[variable + '_2'] = np.zeros(df.shape[0])
    df[variable + '_3'] = np.zeros(df.shape[0])
    df[variable + '_1'][inds1] = split_pt - df[variable][inds1]
    df[variable + '_2'][inds2] = df[variable][inds2] - split_pt
    df[variable + '_3'][inds1] = 1.0
    return df


# Turn a categorical series to a few columns of dummy variables
# each unique value will be a separate column
#
# s - a data series
def get_dummies_column(s):
    vc = s.value_counts()
    names = vc.index
    length = vc.values.shape[0]

    # print names
    column_name = s.name;
    row_num = s.shape[0]
    # print row_num

    data = np.zeros((row_num, length))

    column_names = [''] * (length)
    for i in xrange(length):
        column_names[i] = column_name + '_' + names[i]
        data[:, i] = (s == names[i]).astype(int)

    return pd.DataFrame(data, s.index, columns=column_names)


# Turn a list of categorical series to dummy series, append them,
def process_dummies(data, columns):
    df = data;
    for i in xrange(len(columns)):
        column = columns[i]
        df[column] = df[column].astype(str)
        dummy_series = get_dummies_column(df[column])
        df = pd.concat([df, dummy_series], axis=1)
    return df


# clean up, floor values to 2*p99 by default
def treat_floor(df, names):
    for name in names:
        temp = df[name].quantile(0.99)
        if temp >= 0:
            df[name] = np.minimum(2.0 * temp, df[name])
        else:
            df[name] = np.minimum(0.5 * temp, df[name])
    return df


# clean up, ceiling values to p1*2 by default
def treat_ceiling(df, names):
    for name in names:
        temp = df[name].quantile(0.01)
        if temp > 0:
            df[name] = np.maximum(temp * 0.5, df[name])
        else:
            df[name] = np.maximum(temp * 2.0, df[name])
    return df


# Evaluate output of a logit
# Plot appropriate figures: KS/AUC, score distribution/average score
def evaluate_performance(all_target, predicted, toplot=True, verbose=False):
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    maxind = plb.find(tpr - fpr == ks)

    event_rate = sum(all_target) / 1.0 / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1 - event_rate)
    minind = plb.find(abs(cum_total - event_rate) == min(abs(cum_total - event_rate)))
    if minind.shape[0] > 0:
        minind = minind[0]

    if verbose == True:
        print ('KS=' + str(np.round(ks, 2)) + ', AUC=' + str(np.round(roc_auc, 2)) + ', N=' + str(predicted.shape[0]))
        print (
        'At threshold=' + str(np.round(thresholds[maxind], 3)) + ', TPR=' + str(np.round(tpr[maxind], 2)) + ', ' + str(
            int(np.round(tpr[maxind] * event_rate * all_target.shape[0]))) + ' out of ' + str(
            int(np.round(event_rate * all_target.shape[0]))))
        print (
        'At threshold=' + str(np.round(thresholds[maxind], 3)) + ', FPR=' + str(np.round(fpr[maxind], 2)) + ', ' + str(
            int(np.round(fpr[maxind] * (1.0 - event_rate) * all_target.shape[0]))) + ' out of ' + str(
            int(np.round((1.0 - event_rate) * all_target.shape[0]))))
    recall_ = tpr[maxind] * event_rate * all_target.shape[0] * 1.0 / (event_rate * all_target.shape[0])
    if verbose == True:
        print ('recall= ' + str(np.round(recall_, 2)))
        precision_ = tpr[maxind] * event_rate * all_target.shape[0] * 1.0 / (
            tpr[maxind] * event_rate * all_target.shape[0] * 1.0 + fpr[maxind] * (1.0 - event_rate) * all_target.shape[
                0])
        print ('precision= ' + str(np.round(precision_, 2)))
        f1_score = 2 / (1 / recall_ + 1 / precision_)
        print ('f1_score= ' + str(np.round(f1_score, 2)))

    if toplot:
        # KS plot
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr)
        plt.hold
        plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=2)
        plt.title('KS=' + str(np.round(ks, 2)) + ' AUC=' + str(np.round(roc_auc, 2)), fontsize=20)
        plt.plot([fpr[maxind], fpr[maxind]], [fpr[maxind], tpr[maxind]], linewidth=4, color='r')
        plt.plot([fpr[minind]], [tpr[minind]], 'k.', markersize=10)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False positive', fontsize=20);
        plt.ylabel('True positive', fontsize=20);

        # print ('At threshold=' + str(round(event_rate, 3)))
        # print (str(round(fpr[minind],2)))
        # print (str(int(round(fpr[minind]*(1.0-event_rate)*all_target.shape[0]))))
        # print (str(int(round((1.0-event_rate)*all_target.shape[0]))))


        # Score distribution score
        plt.subplot(1, 3, 2)
        # print predicted.columns
        plt.hist(predicted, bins=20)
        plt.hold
        plt.axvline(x=np.mean(predicted), linestyle='--')
        plt.axvline(x=np.mean(all_target), linestyle='--', color='g')
        plt.title('N=' + str(all_target.shape[0]) + ' Tru=' + str(np.round(np.mean(all_target), 3)) + ' Pred=' + str(
            np.round(np.mean(predicted), 3)), fontsize=20)
        plt.xlabel('Target rate', fontsize=20)
        plt.ylabel('Count', fontsize=20)

        # Score average by percentile
        binnum = 10
        ave_predict = np.zeros((binnum))
        ave_target = np.zeros((binnum))
        indices = np.argsort(predicted)
        binsize = int(np.round(predicted.shape[0] / 1.0 / binnum))
        for i in range(binnum):
            startind = i * binsize
            endind = min(predicted.shape[0], (i + 1) * binsize)
            ave_predict[i] = np.mean(predicted[indices[startind:endind]])
            ave_target[i] = np.mean(all_target[indices[startind:endind]])

        plt.subplot(1, 3, 3)
        plt.plot(ave_predict, 'b.-', label='Prediction', markersize=5)
        plt.hold
        plt.plot(ave_target, 'r.-', label='Truth', markersize=5)
        plt.legend(loc='lower right')
        plt.xlabel('Percentile', fontsize=20)
        plt.ylabel('Target rate', fontsize=20)
        print ('Ave_target: ' + str(ave_target))
        print ('Ave_predicted: ' + str(ave_predict))

    return np.round(ks, 2), np.round(roc_auc, 2)


def evaluate_performance_sim(all_target, predicted, P_label=None, more_eva=0):
    # type: (object, object) -> object
    fpr, tpr, thresholds = roc_curve(all_target, predicted)
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    maxind = plb.find(tpr - fpr == ks)


    thres = np.round(thresholds[maxind], 3)
    event_rate = sum(all_target) / 1.0 / all_target.shape[0]
    cum_total = tpr * event_rate + fpr * (1 - event_rate)

    recall_ = tpr[maxind] * event_rate * all_target.shape[0] * 1.0 / (event_rate * all_target.shape[0])
    precision_ = tpr[maxind] * event_rate * all_target.shape[0] * 1.0 / (
        tpr[maxind] * event_rate * all_target.shape[0] * 1.0 + fpr[maxind] * (1.0 - event_rate) * all_target.shape[0])
    f1_score = 2 / (1 / recall_ + 1 / precision_)

    if more_eva == 1:
        if P_label is not None:

            stat_parity_dev = max(sum(predicted[P_label == 0]) / sum(P_label == 0),
                                  (sum(predicted[P_label == 1]) / sum(P_label == 1))) / min(
                sum(predicted[P_label == 0]) / sum(P_label == 0), (sum(predicted[P_label == 1]) / sum(P_label == 1)))
            stat_parity_sub = abs(
                sum(predicted[P_label == 0]) / sum(P_label == 0) - sum(predicted[P_label == 1]) / sum(P_label == 1))
            #         print stat_parity
            return np.round(ks, 4), np.round(recall_, 4)[0], np.round(precision_, 4)[0], np.round(f1_score, 4)[
                0], np.round(stat_parity_dev, 4), np.round(stat_parity_sub, 4), np.round(thres, 3)
        else:
            return np.round(ks, 4), np.round(recall_, 4)[0], np.round(precision_, 4)[0], np.round(f1_score, 4)[0]
    else:
        if P_label is not None:
            stat_parity = max(sum(predicted[P_label == 0]) / sum(P_label == 0),
                              (sum(predicted[P_label == 1]) / sum(P_label == 1))) / min(
                sum(predicted[P_label == 0]) / sum(P_label == 0), (sum(predicted[P_label == 1]) / sum(P_label == 1)))
            return np.round(ks, 2), np.round(recall_, 2)[0], np.round(precision_, 2)[0], np.round(f1_score, 2)[
                0], np.round(stat_parity, 4)
        else:
            return np.round(ks, 2), np.round(recall_, 2)[0], np.round(precision_, 2)[0], np.round(f1_score, 2)[0]


# Get header row of a file
def get_header(fi):
    f = open(fi, 'r')
    g = csv.reader(f)
    head = g.next()
    head = [x.replace('\xef\xbb\xbf', '') for x in head]
    f.close()
    return head


# Get string for columns to keep to pass to awk
def get_column_string(header, columns):
    ss = '$' + str(header.index(columns[0]) + 1)
    for i in range(1, len(columns)):
        ss = ss + ',$' + str(header.index(columns[i]) + 1)
    return ss


# get dataframe that correspond to a unique field
def get_data(g, currentrow, header, fieldtomatch, tomatch):
    if len(currentrow) == 0:
        return [], []
    index = header.index(fieldtomatch)
    if currentrow[index] > tomatch:
        return [], currentrow
    elif currentrow[index] < tomatch:
        while True:
            try:
                row = g.next()
                currentrow = row
                if row[index] > tomatch:
                    return [], currentrow
                elif row[index] == tomatch:
                    break
            except StopIteration:
                return [], []

    rows = [currentrow]
    while True:
        try:
            row = g.next()
            if row[index] == tomatch:
                rows.append(row)
            else:
                return pd.DataFrame(rows, columns=header), row
        except StopIteration:
            return pd.DataFrame(rows, columns=header), []


# # save an object to a file
# def save_object(obj, filename):
#     with open(filename, 'wb') as output:
#         cPickle.dump(obj, output, -1)
