from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
from scipy.ndimage import shift
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import matplotlib.dates
import seaborn as sns
import numpy as np
import talib, math

#Here the data of the indicators are normalized: atr, ravi.
def norm(a, split):
    a = np.round(a, 5)
    train = (a[:split] - np.min(a[:split])) / (np.max(a[:split]) - np.min(a[:split]))
    max = np.max(a[:split])
    min = np.min(a[:split])
    test = (a[split:] - min) / (max - min)

    b = np.copy(a)
    b[:split] = train
    b[split:] = test
    b = np.round(b, 5)

    return b

"""
HT_DCPERIOD, HT_DCPHASE, ATR indicators are calculated by the TA-Lib library
https://github.com/TA-Lib/ta-lib-python?ysclid=lkjjhzsb50100751365
"""
def ht_(v):
    _ht = talib.HT_DCPERIOD(v)
    _ht = np.nan_to_num(_ht, nan=17.0)
    _ht = shift(_ht, 1, cval=17.0)
    _ht = np.round(_ht, 5)

    return _ht


def htph_(v):
    _htph = talib.HT_DCPHASE(v)
    _htph = np.nan_to_num(_htph, nan=0.0)
    _htph = shift(_htph, 1, cval=0.0)
    _htph = np.round(_htph, 5)

    return _htph


def atr(high, low, close, period, split):
    m = np.nan_to_num(talib.ATR(high, low, close, timeperiod=period))
    m = norm(m, split)

    return m


def ravi(close, open, period, split):
    ma = 0
    mom = (close - open) / (open / 100.0)
    if period > 1:
        ma = np.nan_to_num(talib.EMA(mom, timeperiod=period))
    if period == 1:
        ma = mom

    m = norm(ma, split)

    return m

"""
dataset is created like this:
1. A matrix filled with zeros is created.
2. Atr and ravi are selected from the list aaa and, starting from shift 1, are set into the matrix. 
'hist' is how many elements to take from history.
3. The bbb list data enters the function shifted: ht, htph, df['news1'], df['news2'], df['news3'].
"""

def dataset(x, hist, aaa, bbb):
    hist_ = hist * len(aaa) + len(bbb)
    q = np.ones((x, hist_), dtype=float)
    index_ds = 0
    for fff in aaa:
        sdvig = 1
        for i in range(0, hist):
            q[0:x, index_ds] = shift(fff, sdvig, cval=0.3)
            sdvig += 1
            index_ds += 1

    for fff in bbb:
        q[0:x, index_ds] = fff
        index_ds += 1

    return q


"""
msk - this is where the transition of the forecast from 0 to 1 occurs (the deal is closed)
mask_one -  is where the trade is opened
as a result, we get segments where the start of the deal is marked 1, and the next elements
(in the case of the predicted class 0) and the index where the deal closes are set to 0.
By applying cumsum(), we get each next group more by 1.
"""
def mask(df, msk, label, cs):
    mask_one = (df[label] == 0) & (df[label].shift() != 0)
    df.loc[(df[label] == 0) | msk, cs] = 0
    df.loc[mask_one, cs] = 1
    df[cs] = df[cs].cumsum()

"""
selldd checks if the specified stop loss has been triggered. If it worked, then -stopcs is set and
trades are no longer opened until the class label 1 appears.

Otherwise, set: (opening - closing) / opening.
At the end, a cumulative sum is obtained.
"""
def depoS(df, stopcs, cs, depozit):
    def f(x):
        selldd = (df.loc[x.index, 'High'] - df.loc[x.index[0], 'Open']) / df.loc[x.index[0], 'Open']
        sell_mask = selldd[selldd >= stopcs]
        if len(sell_mask) > 0:
            close = sell_mask.index[0]
            df.loc[close, depozit] = -stopcs
        else:
            df.loc[x.index[-1], depozit] = \
                (df.loc[x.index[0], 'Open'] - df.loc[x.index[-1], 'Open']) / df.loc[x.index[0], 'Open']

    df.groupby(cs)[depozit].apply(f)

    arr = df.loc[~df[depozit].isna(), depozit].values

    df[depozit] = df[depozit].replace(np.nan, 0.0).cumsum() + 1

    return arr


def class_label(ds, y, split, depth=15, tree=150, L=0.01):
    clf = GradientBoostingClassifier(max_depth=depth, max_features="sqrt", n_estimators=tree,
                                     learning_rate=L, random_state=0)
    clf.fit(ds[:split], y)

    return clf.predict(ds)


def stat_balance(aaa, bbb, fff, y, split):
    msq = math.sqrt(252)
    for i in [0, 1]:
        sr_train = round((aaa[i].loc[:split[0], 'cbr'].mean() / aaa[i].loc[:split[0], 'cbr'].std()) * msq,
                         2)
        sr_test = round((aaa[i].loc[split[0]:, 'cbr'].mean() / aaa[i].loc[split[0]:, 'cbr'].std()) * msq,
                        2)

        train = aaa[i].loc[:split[0], bbb[i]]
        test = aaa[i].loc[split[0]:, bbb[i]]

        print(
            '{0} Pearson corr train {1} Pearson corr test {2} Sharp_Ratio_train {3} Sharp_Ratio_test {4} amount of deals {5}'
            .format('this is ' + bbb[i] + ' ---', round(pearsonr(np.arange(len(train)), train)[0], 3),
                    round(pearsonr(np.arange(len(test)), test)[0], 3), sr_train, sr_test, len(aaa[i])))

        print('Train', classification_report(y[:split[1]], fff[i][:split[1]]))
        print('Test', classification_report(y[split[1]:], fff[i][split[1]:]))


def graf_balance(aaa, bbb, ind, il):
    locator = matplotlib.ticker.LinearLocator(12)

    fig, ax = plt.subplots(1, 2)
    fig.subplots_adjust(wspace=0.07, bottom=0.001, top=0.977, left=0.03, right=0.97)

    for i in [0, 1]:
        ax[i].plot(ind[i], aaa[i][bbb[i]], label=bbb[i])
        ax[i].plot(ind[i], aaa[i]['sah'], label='Sell and hold')
        ax[i].xaxis.set_major_locator(locator)
        ax[i].legend()
        ax[i].annotate('Train', xy=(ind[i][il[i]], aaa[i].loc[aaa[i].index[il[i]], bbb[i]]),
                       xycoords='data',
                       bbox=dict(boxstyle='round', fc='none', ec='gray'),
                       xytext=(10, -40), textcoords='offset points', ha='center',
                       arrowprops=dict(arrowstyle='->'))

    fig.autofmt_xdate()
    plt.legend()
    plt.show()


def graf_distribution(arr_no, arr_n):
    fig, ax = plt.subplots(1, 2)
    sns.histplot(arr_no, ax=ax[0], kde=True, color='r')
    sns.histplot(arr_n, ax=ax[1], kde=True, color='r')
    ax[0].set_title('Depozit_no')
    ax[1].set_title('Depozit_n')
    ax[0].axvline(color='blue', linestyle='--')
    ax[1].axvline(color='blue', linestyle='--')

    plt.show()



