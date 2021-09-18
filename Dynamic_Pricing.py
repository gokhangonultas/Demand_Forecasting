#Gerekli Kütüphanelerin Kurulması
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

#Veri Seti Ayarları
pd.set_option('display.max_columns', None)
import time
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')


#Veri Hakkında Kısa Bir Bigi
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Verinin Yüklenmesi

train = pd.read_csv('datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('datasets/demand_forecasting/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('datasets/demand_forecasting/sample_submission.csv')
df = pd.concat([train, test], sort=False)


# EDA(Keşifçi Veri Analizi)

df["date"].min(), df["date"].max()


check_df(train)
check_df(test)
check_df(sample_sub)
check_df(df)

# Satış dağılımı nasıl?
df["sales"].describe([0.10, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99])

# Kaç store var?
df[["store"]].nunique()

# Kaç item var?
df[["item"]].nunique()

# Her store'da eşit sayıda mı eşsiz item var?
df.groupby(["store"])["item"].nunique()

# Peki her store'da eşit sayıda mı sales var?
df.groupby(["store", "item"]).agg({"sales": ["sum"]})

# mağaza-item kırılımında satış istatistikleri
df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.date.dt.dayofyear.describe().T

# FEATURE ENGINEERING
#####################################################

########################
# Date Features Engineers
########################

def create_date_features(df):
    df['month'] = df.date.dt.month  # Hangi ay
    df['day_of_month'] = df.date.dt.day # Ayın hangi günü
    df['day_of_year'] = df.date.dt.dayofyear # yılın hangi günü
    df['week_of_year'] = df.date.dt.weekofyear # yılın hangi haftası
    df['day_of_week'] = df.date.dt.dayofweek #Haftanın hanig günü
    df['year'] = df.date.dt.year  # Hangi yıl
    df["is_wknd"] = df.date.dt.weekday // 4 # Haftasonu mu değil mi
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)  # Ayın başlangıcı mı
    df['is_month_end'] = df.date.dt.is_month_end.astype(int) #Ayın bitişi mi maaş gibi olarak düşünülebilir.
    return df

df = create_date_features(df)

df['day_of_week'] = df['day_of_week'] +1

# Mevsim
df.loc[(df['month'] == 1 ) & (df['month'] == 2 ) & (df['month'] == 12 ), 'season'] = 1
df.loc[(df['month'] > 2 ) & (df['month'] <= 5 ), 'season'] = 2
df.loc[(df['month'] > 5 ) & (df['month'] <= 8 ), 'season'] = 3
df.loc[(df['month'] > 8 ) & (df['month'] <= 11 ), 'season'] = 4
df["season"].unique()
df["month"].unique()
df[df['month'] == 12].count()

# Ayın son günlerine göre

df.loc[(df["day_of_month"] == 31), "fewer_days_than_others"] = 1
df.loc[(df["day_of_month"] == 30), "fewer_days_than_others"] = 1
df.loc[(df["day_of_month"] == 29), "fewer_days_than_others"] = 1
df.loc[(df["day_of_month"] < 29) , "fewer_days_than_others"] = 0

# Yıllara göre satışta önem sıralaması
df.loc[(df["year"] == 2013), "year_imp_on_sale"] = 1
df.loc[(df["year"] == 2014), "year_imp_on_sale"] = 2
df.loc[(df["year"] == 2015), "year_imp_on_sale"] = 3
df.loc[(df["year"] == 2016), "year_imp_on_sale"] = 4
df.loc[(df["year"] == 2017), "year_imp_on_sale"] = 5

# Mağazaların satış rakamlarına göre
df.loc[(df["store"] == 8) | (df["store"] == 2), "sale_amount"] = 3
df.loc[(df["store"] == 3) | (df["store"] == 4), "sale_amount"] = 2
df.loc[(df["store"] == 9) | (df["store"] == 10)| (df["store"] == 1), "sale_amount"] = 2
df.loc[(df["store"] == 5) | (df["store"] == 6)| (df["store"] ==7), "sale_amount"] = 1
df.head()

# Random Noise

def random_noise(dataframe):
    return np.random.normal(scale=1.4, size=(len(dataframe),))

# Lag/Shifted Features
df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# bu değerler 3 ay sonrasını tahmin edilmesi istendiği için 3 aydan öncesini tahmin etmek gerekir .
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 250,364, 546, 728,821])

df[df["sales"].isnull()]


# Rolling Mean Features

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [180,365,450, 546])


# Exponentially Weighted Mean Features

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.99,0.95, 0.9, 0.8, 0.7, 0.5,0.4]
lags = [91, 95,98, 105, 112,150, 180, 270, 365,456, 546, 728,821]

df = ewm_features(df, alphas, lags)

# One-Hot Encoding

df = pd.get_dummies(df, columns=['store', 'item',"season","year_imp_on_sale", 'day_of_week', 'month'])

# Converting sales to log(1+sales)
df['sales'] = np.log1p(df["sales"].values)

#####################################################
# Model

# Custom Cost Function
def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

# Smape i lightgbm özelinde tanımlama yaptık
def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# Time-Based Validation Sets
# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]
# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]
cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]
Y_train = train['sales']
X_train = train[cols]
Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

# LightGBM Model

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape, # yukarıdaki fonksiyonlardan çekiyor sale değişkenleri karşılaştırılması yapıyor
                  verbose_eval=100)
# Custom Coss fonksiyonu oluşturduk hatamızı hesaplıyor lgbm_smape
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))


# Değişken Önem Düzeyleri


def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30)
plot_lgb_importances(model, num=30, plot=True)
# Gain feature kazandırdığı kazanımdır. bölmeden önce ve bölmeden sonraki entropi değişimidir.
lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()

# Final Model
########################
# Sales NA Olmayanlar
train = df.loc[~df.sales.isna()]

Y_train = train['sales']
X_train = train[cols]

test = df.loc[df.sales.isna()]
X_test = test[cols]

# earlystopping rounds ı getirmiyoruz. Çünkü işlem bitti.
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

# LightGBM dataset veriyi oluşturuyoruz
lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


# Submission Sonucunun Kaggle Yüklenmesi
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

smape(np.expm1(test_preds), np.expm1(Y_val))