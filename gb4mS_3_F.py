import numpy as np
import pandas as pd
import qnt

df = pd.read_csv('dataset.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)

stop = 0.05  # stop 5%

x = len(df)
split = int(x * 0.6757)#used for splitting into training and test sets

#y --- labels for the classifier, if the close is less than the open then 0 in other cases 1
y = np.where(df['Close'] < df['Open'], 0, 1)

m_ = qnt.atr(df['High'], df['Low'], df['Close'].values, period=1, split=split)#Average True Range
r_ = qnt.ravi(df['Close'], df['Open'], period=4, split=split)#(close - open) / (open / 100.0) smoothed out EMA
ht = qnt.ht_(df['Close'])#Hilbert Transform - Dominant Cycle Period
htph = qnt.htph_(df['Close'])#Hilbert Transform - Dominant Cycle Phase

tl = [r_, m_]
tln = [ht, htph, df['news1'], df['news2'], df['news3']]

"""
Further, where there are prefixes _n, _no: _n --- macroeconomic indicators are used, _no --- are not used.
q_n, q_no are datasets for training 3 and 1 is how much to take m_, r_ from history starting from shift 1, 
for each instance. ht, htph, news1, news2, news3 are fed into the function with shift.
"""
q_n = qnt.dataset(x, 3, tl, tln)
q_no = qnt.dataset(x, 1, tl, tln[:2])
#Here we get the predicted classes by the GradientBoostingClassifier model.
yALL_n = qnt.class_label(q_n, y[:split], split, depth=5, tree=30, L=0.01)
yALL_no = qnt.class_label(q_no, y[:split], split, depth=3, tree=30, L=0.01)
"""
'cs_n', 'cs_no' - columns for grouping transactions into groups so as not to use a cycle
label_n, label_no - predicted classes, sah - Sell and hold
"""
df[['cs_n', 'cs_no', 'Depozit_n', 'Depozit_no']] = np.nan
df = df.assign(label_n=yALL_n, label_no=yALL_no, sah=df['Open'].pct_change(-1).cumsum() + 1)
#mask_n, mask_no this is where the transition of the forecast from 0 to 1 occurs (the deal is closed)
mask_n = (df['label_n'] == 1) & (df['label_n'].shift() == 0) | (df.index == df.index[-1])
mask_no = (df['label_no'] == 1) & (df['label_no'].shift() == 0) | (df.index == df.index[-1])
#in the mask function, the cs_n and cs_no columns are marked into groups. More detailed description in the included file qnt.
qnt.mask(df, mask_n, 'label_n', 'cs_n')
qnt.mask(df, mask_no, 'label_no', 'cs_no')
"""
Here the balance of the deposit for transactions is calculated. Arrays arr_n, arr_no are obtained to display 
the histogram of distributions by deals. More detailed description in the included file qnt.
"""
arr_n = qnt.depoS(df, stop, 'cs_n', 'Depozit_n')
arr_no = qnt.depoS(df, stop, 'cs_no', 'Depozit_no')

df_n = df.loc[mask_n, ['sah', 'Depozit_n']].copy()
df_no = df.loc[mask_no, ['sah', 'Depozit_no']].copy()
#'cbr' is used to calculate Sharp Ratio.
df_n['cbr'] = df_n['Depozit_n'].pct_change().replace(np.nan, 0.0)
df_no['cbr'] = df_no['Depozit_no'].pct_change().replace(np.nan, 0.0)

aaa = [df_no, df_n]
bbb = ['Depozit_no', 'Depozit_n']
fff = [yALL_no, yALL_n]
"""
Statistics are displayed here: balance Pearson correlation, Sharpe Ratio,
number of transactions, precision, recall, and so on.
"""
qnt.stat_balance(aaa, bbb, fff, y, [df.index[split], split])

ind = [df_no.index.format(), df_n.index.format()]
# il --- used for splitting training and test data
il = [np.where(df_no.index >= df.index[split])[0][0], np.where(df_n.index >= df.index[split])[0][0]]
"""
Here, graphs of balances are drawn: on the left without the addition 
of macroeconomic indicators, on the right with the addition. Orange: 'Sell and hold'.
"""
qnt.graf_balance(aaa, bbb, ind, il)

# Drawing trade distribution histograms (trades for the test segment after the split).
qnt.graf_distribution(arr_no[il[0]:], arr_n[il[1]:])









