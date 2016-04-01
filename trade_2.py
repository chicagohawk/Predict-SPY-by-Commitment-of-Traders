'commitment of traders strategy, \
 historical data: quandl google finance and CFTC financial (futures-only) \
 realtime data, 3.30pm Friday, \
 http://www.cftc.gov/dea/futures/financial_lf.htm \
 historical data, traders in financial futures, futures only, 2015 text, \
 http://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm'


import numpy as np
import pandas as pd
from datetime import datetime as dt
import os
from pdb import set_trace
import re
from matplotlib import pyplot as plt
import subprocess, time

# ------------------- PRE-PROCESSING ------------------

HOME = '/home/voila/Documents/2014GRAD/QuantCraft/CFTC/'

PAIR = open(HOME+'data/PAIR')
PAIR = PAIR.readlines()
PAIR0 = []
for p in PAIR:
    p = p.split('|')
    PAIR0.append( [p[0], p[1]] )
PAIR = PAIR0

# relist column: date, code, open_interest, long, short
print ('Loading COT')
#      date  code openint  dealder  manager  hedge-fund  other-report  non-report
col = [2,    3,   7,       8,9,     11,12,   14,15,      17,18,        22,23]
#      0     1    2        3 4      5  6     7  8        9  10         11 12

print('-'*40)
print('Remember to update historical data when turn to 2016!')
print('-'*40)
years = ['2010','2011','2012','2013','2014','2015','2016']
for j in range(len(years)):
    COT = open(HOME+'data/COT_'+years[j])
    COT = COT.readlines()
    header = COT[0].split(',')
    # COT table validality check
    assert(header[col[0]][1:-1] == 'Report_Date_as_YYYY-MM-DD' or
           header[col[0]][1:-1] == 'Report_Date_as_MM_DD_YYYY')
    assert(header[col[1]][1:-1] == 'CFTC_Contract_Market_Code')
    assert(header[col[2]][1:-1] == 'Open_Interest_All')
    assert(header[col[3]][1:-1] == 'Dealer_Positions_Long_All')
    assert(header[col[4]][1:-1] == 'Dealer_Positions_Short_All')
    assert(header[col[5]][1:-1] == 'Asset_Mgr_Positions_Long_All')
    assert(header[col[6]][1:-1] == 'Asset_Mgr_Positions_Short_All')
    assert(header[col[7]][1:-1] == 'Lev_Money_Positions_Long_All')
    assert(header[col[8]][1:-1] == 'Lev_Money_Positions_Short_All')
    assert(header[col[9]][1:-1] == 'Other_Rept_Positions_Long_All')
    assert(header[col[10]][1:-1] == 'Other_Rept_Positions_Short_All')
    assert(header[col[11]][1:-1] == 'NonRept_Positions_Long_All')
    assert(header[col[12]][1:-1] == 'NonRept_Positions_Short_All')
    if 'COT0' not in dir():
        COT0 = COT[1:][::-1]
    else:
        COT0 += COT[1:][::-1]
COT = COT0

# input signals
# data  openint  mgr-long-short  hf-long-short  rep-long-short  retail-long-short  dealer-long-short
# 0     1        2 3             4 5            6 7             8 9                10 11
SIGNAL = ()
for prod in range(len(PAIR)):
    input_prod = []
    for j in range(len(COT)):
        cotline = COT[j].split(',')
        if cotline[col[1]].strip() != PAIR[prod][0]:
            continue
        else:
            input_prod.append( 
            [ cotline[col[0]], cotline[col[2]], cotline[col[5]], cotline[col[6]], 
              cotline[col[7]], cotline[col[8]], cotline[col[9]], cotline[col[10]],
              cotline[col[11]], cotline[col[12]], cotline[col[3]], cotline[col[4]] ])
    SIGNAL += (input_prod,)

# load output files
print('Loading prices')
PRICE0 = {}
for prod in range(len(PAIR)):
    output_prod = []
    underlie = PAIR[prod][1]
    fn = HOME+'data/'+underlie+'.csv'
    PRICE = open(fn)
    PRICE = PRICE.readlines()
    header = PRICE[0].split(',')
    # PRICE table validality check
    assert(header[0] == 'Date')
    assert(header[1] == 'Open')
    assert(header[2] == 'High')
    assert(header[3] == 'Low')
    assert(header[4] == 'Close')
    assert(header[5][:-1] == 'Volume')
    if PRICE0.get(underlie) is None:
        PRICE0[underlie] = PRICE[1:][::-1]
PRICE = PRICE0

PRICE0 = {}
for prod in PRICE.keys():
    output_prod = []
    for j in range(len(PRICE[prod])):
        output_prod += [ PRICE[prod][j].split(',') ]
    PRICE0[prod] = output_prod
PRICE = PRICE0

# transpose SIGNAL and PRICE
print('Transposing data')
SIGNAL0 = []
for i in range(len(PAIR)):
    dates, openint = [], []
    mgr_lon,mgr_sht,hf_lon,hf_sht,orp_lon,orp_sht,nrp_lon,nrp_sht,dlr_lon,dlr_sht = \
    [],     [],     [],    [],    [],     [],     [],     [],     [],     []
    retail_lon, retail_sht = [], []
    for j in range(len(SIGNAL[i])):
        dates.append(SIGNAL[i][j][0])
        openint.append(float(SIGNAL[i][j][1]))
        mgr_lon.append( float(SIGNAL[i][j][2]) )
        mgr_sht.append( float(SIGNAL[i][j][3]) )
        hf_lon.append( float(SIGNAL[i][j][4]) )
        hf_sht.append( float(SIGNAL[i][j][5]) )
        orp_lon.append( float(SIGNAL[i][j][6]) )
        orp_sht.append( float(SIGNAL[i][j][7])  )
        nrp_lon.append( float(SIGNAL[i][j][8]) )
        nrp_sht.append( float(SIGNAL[i][j][9])  )
        dlr_lon.append( float(SIGNAL[i][j][10]) )
        dlr_sht.append( float(SIGNAL[i][j][11])  )
    dates = pd.Series(dates)
    df = pd.DataFrame(data={
         'DATE':dates,'OPENINT':openint,
         'MGRL':mgr_lon,'MGRS':mgr_sht,
         'HFL':hf_lon,  'HFS':hf_sht,
         'ORPL':orp_lon, 'ORPS':orp_sht,
         'NRPL':nrp_lon, 'NRPS':nrp_sht,
         'DLRL':dlr_lon, 'DLRS':dlr_sht })
    df = df.set_index('DATE')
    SIGNAL0.append(df)
SIGNAL = SIGNAL0

PRICE0 = {}
for prod in PRICE.keys():
    dates, opn, high, low, close, volume = [],[],[],[],[],[]
    for j in range(len(PRICE[prod])):
        dates.append(PRICE[prod][j][0])
        try:
            opn.append(float(PRICE[prod][j][1]))
        except ValueError:
            opn.append(np.nan)
        try: 
            high.append(float(PRICE[prod][j][2]))
        except ValueError:
            high.append(np.nan)
        try:
            low.append(float(PRICE[prod][j][3]))
        except ValueError:
            low.append(np.nan)
        try:
            close.append(float(PRICE[prod][j][4]))
        except ValueError:
            close.append(np.nan)
        try:
            volume.append(float(PRICE[prod][j][5][:-1]))
        except ValueError:
            volume.append(np.nan)
    dates = pd.Series(dates)
    df = pd.DataFrame(data={'DATE':dates,'OPEN':opn,'HIGH':high,
                            'LOW':low,'CLOSE':close,'VOLUME':volume})
    df = df.set_index('DATE')
    PRICE0[prod] = df
PRICE = PRICE0

# calculate price series momentum (10-day), RSI (14-day), and MFI (14-days)
print('Calculate secondary series')
for prod in PRICE.keys():
    previous = PRICE[prod].shift(periods=1)
    # RSI
    PRICE[prod]['UPMOVE'] = \
        (PRICE[prod]['CLOSE'] > previous['CLOSE']) * \
        (PRICE[prod]['CLOSE'] - previous['CLOSE'])
    PRICE[prod]['DOWNMOVE'] = \
        (PRICE[prod]['CLOSE'] < previous['CLOSE']) * \
        (previous['CLOSE'] - PRICE[prod]['CLOSE'])
    U = pd.ewma(PRICE[prod]['UPMOVE'],com=6.5)
    D = pd.ewma(PRICE[prod]['DOWNMOVE'],com=6.5)
    RS = U/D
    PRICE[prod]['RSI'] = 100-100./(1+RS)
    PRICE[prod].drop('UPMOVE',axis=1,inplace=True)
    PRICE[prod].drop('DOWNMOVE',axis=1,inplace=True)
    # MFI
    PRICE[prod]['TP'] = (PRICE[prod]['HIGH']+PRICE[prod]['LOW']+PRICE[prod]['CLOSE']) / 3.
    PRICE[prod]['POSFLOW'] = \
        (PRICE[prod]['CLOSE'] > previous['CLOSE']) * \
        PRICE[prod]['TP'] * PRICE[prod]['VOLUME']
    PRICE[prod]['NEGFLOW'] = \
        (PRICE[prod]['CLOSE'] < previous['CLOSE']) * \
        PRICE[prod]['TP'] * PRICE[prod]['VOLUME']
    POS = pd.ewma(PRICE[prod]['POSFLOW'],com=6.5)
    NEG = pd.ewma(PRICE[prod]['NEGFLOW'],com=6.5)
    PRICE[prod]['MFI'] = 100 - 100./(1+POS/NEG)
    PRICE[prod].drop('TP',axis=1,inplace=True)
    PRICE[prod].drop('POSFLOW',axis=1,inplace=True)
    PRICE[prod].drop('NEGFLOW',axis=1,inplace=True)
    # momentum
    previous = PRICE[prod].shift(periods=10)
    PRICE[prod]['MOM'] = PRICE[prod]['CLOSE'] - previous['CLOSE']

# convert to weekly price
print('Convert to weekly data')
PRICE_CP = PRICE.copy()
for prod in PRICE.keys():
    week_list = []
    for j in range(len(PRICE[prod])):
        date = dt.strptime(PRICE[prod].index[j],'%Y-%m-%d').isocalendar()
        week_list.append(date[1])
    # last week day
    week_shift = week_list[1:] + [week_list[-1]]
    last_week_day = np.array(week_shift,dtype=int)-np.array(week_list,dtype=int)
    last_week_day = pd.Series(last_week_day, index=PRICE[prod].index)
    PRICE[prod] = PRICE[prod][last_week_day==1]
    yearweek = []
    for j in range(len(PRICE[prod])):
        date = dt.strptime(PRICE[prod].index[j],'%Y-%m-%d').isocalendar()
        yearweek.append(str(date[0])+'_'+str(date[1]))
    PRICE[prod].is_copy = False
    PRICE[prod]['YEARWEEK'] = pd.Series(yearweek, index=PRICE[prod].index)
    PRICE[prod].drop('HIGH',axis=1,inplace=True)
    PRICE[prod].drop('LOW',axis=1,inplace=True)
    PRICE[prod].drop('OPEN',axis=1,inplace=True)
    PRICE[prod].drop('VOLUME',axis=1,inplace=True)

for prod in PRICE_CP.keys():
    yearweek = []
    for j in range(len(PRICE_CP[prod])):
        date = dt.strptime(PRICE_CP[prod].index[j],'%Y-%m-%d').isocalendar()
        yearweek.append(str(date[0])+'_'+str(date[1]))
    PRICE_CP[prod]['YEARWEEK'] = pd.Series(yearweek, index=PRICE_CP[prod].index)
    PRICE_CP[prod].drop('RSI',axis=1,inplace=True)
    PRICE_CP[prod].drop('MFI',axis=1,inplace=True)
    PRICE_CP[prod].drop('HIGH',axis=1,inplace=True)
    PRICE_CP[prod].drop('LOW',axis=1,inplace=True)
    PRICE_CP[prod].drop('OPEN',axis=1,inplace=True)
   

# match PRICE (Friday close) with SIGNAL
for i in range(len(SIGNAL)):
    yearweek = []
    for j in range(len(SIGNAL[i])):
        date = dt.strptime(SIGNAL[i].index[j],'%Y-%m-%d').isocalendar()
        yearweek.append(str(date[0])+'_'+str(date[1]))
    SIGNAL[i]['YEARWEEK'] = pd.Series(yearweek, index=SIGNAL[i].index)

for i in range(len(SIGNAL)):
    SIGNAL[i] = SIGNAL[i].set_index('YEARWEEK')
for prod in PRICE.keys():
    PRICE[prod] = PRICE[prod].set_index('YEARWEEK')

for i in range(len(SIGNAL)):
    SIGNAL[i] = SIGNAL[i].join(PRICE[PAIR[i][1]], how='inner')

# calculate COT spread
for i in range(len(SIGNAL)):
    for player in ['MGR','HF','ORP','NRP','DLR']:
        # manager spread and d_spread
        SIGNAL[i][player+'_SPRD'] = SIGNAL[i][player+'L'] - SIGNAL[i][player+'S']
        SIGNAL[i][player+'_DSPRD'] = SIGNAL[i][player+'_SPRD']  \
                                   - SIGNAL[i][player+'_SPRD'].shift(periods=1)
        SIGNAL[i] = SIGNAL[i][ - SIGNAL[i][player+'_SPRD'].apply(np.isnan) ]    # remove nan entries


# ======================== POST-PROCESSING, SPY ============================
# ------------------------    Dow Jones group   ----------------------------
prod = 0                      # Dow Jones

w_DOW_a = 0.75    # other-reportable spread
value = 2. * SIGNAL[prod]['ORP_SPRD'] / (SIGNAL[prod]['ORPL']+SIGNAL[prod]['ORPS'])
pos = value > ( value.mean() + 1. ) 
neg = ( value < ( value.mean() - .5 ) ) & ( value > ( value.mean() - 1.2 ) )
DOW_a = ((pos*1.) - (neg*1.)) * w_DOW_a
plt.plot(value[-15:], color='green')

del pos, neg
w_DOW_b = 0.75    # other-reportable dspread
value = 2. * SIGNAL[prod]['ORP_DSPRD'] / (SIGNAL[prod]['ORPL']+SIGNAL[prod]['ORPS'])
pos = value > .5
neg = value < -.5
DOW_b = ((pos*1.) - (neg*1.)) * w_DOW_b
plt.plot(value[-15:], color='blue')

DOW = DOW_a + DOW_b

# ------------------------    S&P 500 group   ----------------------------
prod = 1                      # S&P 500

del pos, neg
w_SP_a = .75    # other-reportable spread
value = 2. * SIGNAL[prod]['ORP_SPRD'] / (SIGNAL[prod]['ORPL']+SIGNAL[prod]['ORPS'])
pos = value > ( value.mean() + .6 )
neg = ( value < ( value.mean() + .0 ) ) & ( value > ( value.mean() - .2 ) ) # questionable
SP_a = (pos*1.)* w_SP_a - (neg*1.) * w_SP_a
plt.plot(value[-15:], color='red', linestyle='--')

del pos, neg
w_SP_b = .75   # non-reportable spread
value = 2. * SIGNAL[prod]['NRP_SPRD'] / (SIGNAL[prod]['NRPL']+SIGNAL[prod]['NRPS'])
neg = value > ( value.mean() + .5 )
pos = value < ( value.mean() - .8 )
SP_b = - (neg*1.) * w_SP_b + (pos*1.*1.) * w_SP_b
plt.plot(-value[-15:], color='red')

del neg
w_SP_c = 1.    # non-reportable dspread
value = 2. * SIGNAL[prod]['NRP_DSPRD'] / (SIGNAL[prod]['NRPL'] + SIGNAL[prod]['NRPS'])
pos = (value < -.15) & (value > -.4)
neg = (value > .05) & ( value < .1)
SP_c = (pos*1.*1.) * w_SP_c - (neg*1.) * w_SP_c
plt.plot(-value[-15:], color='red')

SP = SP_a + SP_b + SP_c

# ------------------------    NASDAQ group   ----------------------------
prod = 2                      # NASDAQ, no significant signal found

# ===========================  score committee ==============================
INDICATOR = DOW + SP

plt.plot((PRICE['SPY']['CLOSE'][-15:]-190)/20., color='black')

print('-'*20)

if INDICATOR.iloc[-1] >= 1.:
    print('Buy')
elif INDICATOR.iloc[-1] <= 1.:
    print('Sell')
else:
    print('Hold')
print('-'*20)

print('Indicator series')
print(INDICATOR[-20:])



