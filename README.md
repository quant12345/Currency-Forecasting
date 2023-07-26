# Currency-Forecasting
![Cumulative balance charts](https://github.com/quant12345/Currency-Forecasting/blob/main/balance.jpg)
   When forecasting in finance, derivatives of the price itself are usually used. Sometimes this has some effect, the improvements are barely noticeable, if not accidental. Famous expression: 'garbage in the input garbage out. You can try to look for signs not in the price itself. Central banks, various bureaus produce numerous macroeconomic indicators. I used them
as additional features for forecasting currencies.

  For each currency area, more than 700 types of data were obtained (for each currency a couple of about 1500). I tried to reduce the number of features using feature importance
based on a random forest, the importance of permutations, etc. This did not give the desired effect. Probably because there is little information in the data. Then I started to add one feature to the main dataset and see if there are any improvements in the balance curve. So
Thus, the number of signs was reduced tenfold. This job took a lot time. Then i began to cycle through these selected features adding three to the main dataset or applied a genetic algorithm to find the best combination. A variant with a genetic algorithm found the best combinations with a large number of features and over time it turned out that it was retraining (overfitting).
    In the above chart, the result obtained for EURUSD since 1991 (only sales).Sell and hold in orange. On the left, the model trained without adding macroeconomic indicators ('Depozit_no'), on the right with adding ('Depozit_n'). The model was trained in January 2020 and the settings have not changed since then. Classifier was used for classification: GradientBoostingClassifier. Signs news 1, 2, 3 do not plan to disclose.
  
To reproduce the code, the following libraries are needed: 
numpy, pandas, scipy, sklearn, matplotlib, seaborn, talib. You also need an include file: qnt.

By code:
1. The dataset.csv file is read, in which EURUSD daily quotes and three selected macroeconomic indicators (these are their momentums shifted by the required amount and
split-normalized). Example: if the data is dated October 1st and becomes available on November 15th, then it is shifted by about 45 days.
2. The price series is converted into a series of technical analysis indicators. This data+
macroeconomic indicators create a training dataset.
3.Based on the classifier labels obtained, balance curves are built in
dataframes df_n, df_no.
4. Various indicators, a graph of balances and distributions of transactions are displayed.

![distribution of deals](https://github.com/quant12345/Currency-Forecasting/blob/main/distribution.jpg)
```
import scipy.stats

arr_no_ = arr_no[il[0]:]
arr_n_ = arr_n[il[1]:]

def ep(N, n):
    p = n/N
    sigma = math.sqrt(p * (1 - p)/N)

    return p, sigma

def a_b_statistic(N_A, n_a, N_B, n_b):
    p_A, sigma_A = ep(N_A, n_a)
    p_B, sigma_B = ep(N_B, n_b)

    return (p_B - p_A)/math.sqrt(sigma_A ** 2 + sigma_B ** 2)


z_csore = a_b_statistic(len(arr_n_), (arr_n_ > 0).sum(), len(arr_no_), (arr_no_ > 0).sum())
p_value = scipy.stats.norm.sf(abs(z_csore))

print('z_csore : ' + str(z_csore), 'p value : ' + str(p_value))

lno = round(arr_no_[arr_no_ < 0].sum(), 3)#amount of loss
pno = round(arr_no_[arr_no_ > 0].sum(), 3)#amount of profits
ano = round(abs(pno/lno), 3)#the ratio of the amount of profits to the amount of losses
ln = round(arr_n_[arr_n_ < 0].sum(), 3)
pn = round(arr_n_[arr_n_ > 0].sum(), 3)
an = round(abs(pn/ln), 3)

print('amount of loss no_ {0} amount of profits no_ {1} attitude no {2} amount of loss n_ '
      '{3} amount of profits n_ {4} attitude n {5}'.format(lno, pno, ano, ln, pn, an))


print('arr_n_ > 0  amount {0} arr_n_ < 0 amount {1} arr_no_ > 0 amount {2} arr_no_ < 0 amount {3}'.
      format((arr_n_ > 0).sum(), (arr_n_ < 0).sum(), (arr_no_ > 0).sum(), (arr_no_ < 0).sum()))
```
You can also calculate A / B - test. Which shows that there is no significant difference.
```
z_csore : 0.29180281268336133 p value : 0.3852186972909527
amount of loss no_ -0.49 amount of profits no_ 0.77 attitude no 1.571 amount of loss n_ -0.597 amount of profits n_ 1.096 attitude n 1.836
arr_n_ > 0  amount 162 arr_n_ < 0 amount 100 arr_no_ > 0 amount 169 arr_no_ < 0 amount 99
```
