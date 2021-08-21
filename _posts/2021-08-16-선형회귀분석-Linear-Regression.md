

<br>

예측분석의 목표 : 트레이닝 데이터를 학습시켜 성능이 우수한 모델을 만들고, 만들어진 모델로 새로운 값(새로운 독립변수)에 대한 결과를 예측하고자 하는 것.

<br>

# **회귀분석**  

하나의 독립변수가 종속변수에 영향을 미치는 지를 파악해 인과관계를 추론하는 분석방법이다.  
변수 사이의 관계를 알아보고 독립변수의 값으로 종속변수의 값을 예측할 수도 있다.  

회귀의 기본 원리는 선형회귀모델의 직선과 실제값 사이의 차이인 잔차(error, cost, lost)를 최소화 시키는 것이다.
즉, 최적의 추세선이란 잔차제곱합이 최소가 되는 선이라고 할 수 있다.
최적의 추세선을 구해 예측과 분류를 수행한다.  

독립변수 X로 종속변수 Y를 예측하기 위한 추세선(일차방정식)
> y = wx + b

w : 기울기(weight(가중치))  
b : 절편(bias(편향))


cost가 최소인 추세선을 구하는 과정을 학습(train)이라고 하며 학습에서 사용되는 알고리즘이 경사하강법이다.

<br>

## 최소제곱법  

최소제곱법을 이용해 일차방정식 w(기울기)와 b(절편)


```py
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

x = np.array([0,1,2,3])
y = np.array([-1,0.3,0.9,2.2])

plt.plot(x,y)
plt.grid(True)
plt.show()

A = np.vstack([x, np.ones(len(x))]).T  #x 인수를 행렬로 받기 때문에 행렬로 구성
print(A)
print()

#최소제곱합을 구하는 라이브러리
import numpy.linalg as lin  

print(lin.lstsq(A, y))
print()

w, b = lin.lstsq(A, y)[0]
print(w, b)
print()
#y = wx + b

plt.plot(x, y, 'o', label = '실제값', markersize = 10)
plt.plot(x, w * x + b, 'r', label = '최적화된 추세선')
plt.legend()
plt.show()

yhat = w * 2 + b
print(yhat)
```

<br>

# **선형회귀 Linear Regression**

독립변수와 종속변수와의 선형 상관관계를 모델링하는 방법이다. 
두 변수는 상관관계가 있어야 하며 인과관계가 성립해야 한다. 

독립변수 : 연속형  
종속변수 : 연속형  

기본 충족 조건 : 선형성, 잔차정규성, 독립성, 등분산성, 다중공선성

<br>

# **단순선형회귀 분석 (Simple Linear Regression)**

<br>

### 데이터 생성 : `make_regression()`

`make_regression()` 함수를 사용하여 샘플을 생성한다.  
Scikit-learn에서 feature의 형태를 2차원 데이터(행렬)로 학습하기 때문에 feature 데이터를 2차원으로 변형해주어야 한다.

```py
import statsmodels.api as sm
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd

np.random.seed(12)

x, y, coef = make_regression(n_samples=50, n_features=1, bias=100.0, coef = True)

print(x)    #sklearn에서 x는 matrix
print(y)    #y는 vector 형식
print(coef) #가중치 w
print()

y_pred = 89.47430739278907 * 0.64076111 + 100
print('예측값:', y_pred)
print()

y_pred = 89.47430739278907 * -1.70073563 + 100
print('예측값:', y_pred)
print()

new_x = 1.23
y_pred_new = 89.47430739278907 * new_x + 100
print('예측값:', y_pred_new)
```

<br>

## 모델 생성

<br>

### 방법 1 : Scikit-learn의 `LinearRegression()` 함수 사용

```py
xx = x
yy = y

from sklearn.linear_model import LinearRegression

#모델 객체 생성
model = LinearRegression()
print(model)

#학습
fit_model = model.fit(xx, yy)
print(fit_model.coef_)
print(fit_model.intercept_)

#예측값 확인
y_pred = fit_model.predict(xx[[0]])
print('예측값:', y_pred)

#새로운 값으로 확인
new_x = 1.23
y_pred_new = fit_model.predict([[new_x]])  #matrix로 입력
print('예측값:', y_pred_new)
```

<br>

### 방법 2 : `ols()` 사용   

Scikit-learn에서는 행렬값을 사용했으나, ols에서는 1차원 데이터를 사용한다.  
앞서 사용하던 2차원으로 된 데이터를 차원축소 해주어야 한다.  


```py
import statsmodels.formula.api as smf

print(xx.shape) #2차원  

x1 = xx.flatten()
print(x1.shape)
print(x1)

y1 = yy
print(y1)

data = np.array([x1, y1])
df = pd.DataFrame(data.T)
print(df.head(3))

model2 = smf.ols(formula = 'y1 ~ x1', data=df).fit()
print(model2.summary())
print()

#예측
print(x1[:2])
print(y1[:2])
print()

new_df = pd.DataFrame({'x1':[-1.70073563, -0.67794537]})  #DataFrame으로 학습시켰으므로 DF형식으로 데이터 입력  
new_pred = model2.predict(new_df)
print('예측값:', new_pred)
print()

new2_df = pd.DataFrame({'x1':[1.23, -3.21]}) 
new2_pred = model2.predict(new2_df)
print('예측값:', new2_pred)
```

<br>

### 방법 3 : Scipy stats의 `linregress()` 함수 사용  


```py
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_iq = pd.read_csv('/content/drive/MyDrive/testdata/score_iq.csv')
print(score_iq.head(3))
print(score_iq.describe())
print(score_iq.corr())

#iq가 score에 영향을 준다고 가정한다.
x = score_iq.iq
y = score_iq.score

#두 변수 간의 상관관계 확인
print(np.corrcoef(x,y))
print()

#그래프
plt.scatter(x, y)
plt.xlabel('iq')
plt.ylabel('score')
plt.show()

model = stats.linregress(x,y)

print(model)
print('기울기:', model.slope)
print('절편:', model.intercept)
print('설명력:', model.rvalue**2)
print('p-value:', model.pvalue)
print()
```

pvalue=2.8476895206683644e-50로 인과관계가 존재한다고 할 수 있다. 
회귀모델은 적합하다. 

```py
#예측
new_x = 125
ypred = model.slope * new_x + model.intercept
print('예측값:', ypred)
```
`predict()` 함수를 지원하지 않는다. 대신 Numpy의 `polyval()` 함수를 사용한다.



```py
#전체값 예측
print('함수예측:', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'])))
print()

#새로운 값 예측
new_df = pd.DataFrame({'iq':[55,66,77,88,155]})
print('함수예측:', np.polyval([model.slope, model.intercept], new_df))
```

<br>

## 👉 **`ols()` 결과 해석하기** 


```py
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/content/drive/MyDrive/testdata/drinking_water.csv')
print(df.head(3))
print(df.corr())
print()

model = smf.ols(formula='만족도 ~ 적절성', data=df).fit()
print(model.summary())
```

> Prob (F-statistic):  2.24e-52  

F값에 의해 만들어진 p-value이다. 모형의 적합도를 판단하기 위해 사용한다.  
p-value < 0.05 이면 해당 회귀모델을 유의하다고 할 수 있다.  (독립변수와 종속변수 사이에 인과관계가 존재한다.)

> Intercept  0.7789 -> 절편   
적절성     0.7393 -> 기울기, β  

기울기의 각도가 얼마인지는 상관이 없다. 다만 기울기가 0이어서는 안된다.  0이면 독립변수가 아무리 변화해도 종속변수가 변하지 않기 때문이다.  

가설  
H0 : 기울기(β) = 0  
H1 : 기울기(β) != 0  

표본오차 t, p로 기울기 값이 유의한지 판단한다.

> std err  0.038

표준오차 : 표본평균들의 표준편차
모집단이랑 얼마나 차이가 나는 가에 대한 값이다. 즉, 모집단 평균과 표본평균과의 차이이다.  

표준오차가 작을 수록 모집단에 유사하다.   

표본오차 t, p로 기울기 값이 유의한지 판단할 수 있다.

> t  19.340  

t값  = 기울기 / 표준오차    
19.340 = 0.7393 /  0.038   

t의 제곱 = F-statistic: (F값)     
19.340**2 = 374.0  

t값을 통해서 F값 산출되며 F값 통해서 p-value 산출된다.  
t값과 p값은 반비례관계이다.

> P>|t|   0.000  

각각의 독립변수에 대한 p값이다.  
독립변수의 적절성, 각각의 독립변수가 유의한지 판단하는 값이라고 할 수 있다.    
p < 0.05이므로 해당 독립변수가 유의하다고 할 수 있다.

> [0.025   &emsp;   0.975]  

0.664   0.815  => 신뢰구간에 해당하는 값이다.

> R-squared:   0.588     

결정계수, 모델의 설명력.  
독립변수가 종속변수의 분산을 얼마나 잘 설명할 수 있는가에 대한 값이다.     
결정계수는 0에서 1사이의 값을 가지며 높을 수록 모델의 설명력이 높다는 의미이다.  
0.588이면 꽤 높은편이다.  
현장에서는 0.20 ~ 0.15 (20% ~ 15%) 정도면 설명력이 있다고 본다.  


회귀직선의 적합도(goodness-of-fit)를 평가하거나 종속변수에 대한 독립변수들의 설명력을 알고자 할 때 이용한다.  
설명된 분산은 종속변수의 분산과 독립변수가 나타내는 분산의 교집합이다.  교집합이 아닌부분은 잔차(error)의 분산이다.  
설명력이 높을 수록 교집합이 커지며 독립변수가 종속변수의 많은 부분을 설명한다고 할 수 있다.  


결정계수 = 설명된 분산값 / 종속변수의 전체분산  

상관계수 R을 제곱하거나 혹은 1 - SSE / SST 로 구할 수 있다.

SSE = 모든 오차에서 추정치를 뺀 값 제곱합.  
SSR = 추정치에서 평균을 뺀 값 제곱합.  
SST = 오차에서 평균을 뺀값 제곱합.  SSE + SSR  

 

독립변수가 하나일때는 결정계수로 확하고 독립변수가 두 개 이상일 때는 수정된 결정계수(Adj. R-squared:  0.586)를 사용하여 판단한다.    

결정계수는 독립변수 개수가 늘어날수록 그 값이 커지게 된다. 따라서 종속변수의 분산을 설명해 주지 못하는 변수가 모형에 추가된다고 하더라도 결정계수값이 커질 수 있다.  
이러한 문제를 보정한 것이 수정된 결정계수이다. 수정결정계수는 표본의 크기와 독립변수의 수를 고려하여 계산한다.  

표본의 크기가 200개 이상일 때는 두 결정계수의 차이가 미미하기 때문에 표본이 200개 미만일 때는 반드시 수정결정계수를 고려해야 함.  



> Durbin-Watson:  2.185   

잔차의 독립성 검정.  자기 상관관계를 판단한다.  
0~4값을 가지며  2에 가까울 수록 독립성을 성립한다고 본다.  즉, 2에 가까울 수록 자기상관이 없다.  

주로 시계열 데이터 등에서 패턴이 반복되는 현상 -> 자기상관이 발생한다.

> Skew:   -0.328  

왜도.  
음수면 왼쪽으로 꼬리긴 분포모양, 양수면 오른쪽으로 꼬리가 긴 분포모양을 가진다.

> Kurtosis: 4.012  

첨도.
음수면 분포가 완만, 양수면 분포가 뾰족한 모양이다.


```py
print('coef:\n', model.params)
print('\nr-squared:\n', model.rsquared)
print('\np-value:\n', model.pvalues)
#print('\npred values:\n', model.predict())
print('\npred values:\n', df.적절성[0], model.predict()[0])

#새로운 값 예측
print(df.적절성[:5].values)

new_df = pd.DataFrame({'적절성':[4, 3, 4, 2, 7]})
new_pred = model.predict(new_df)
print(new_pred)

#시각화
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

plt.scatter(df.적절성, df.만족도)
slope, intercept = np.polyfit(df.적절성, df.만족도, 1)
plt.plot(df.적절성, df.적절성 * slope + intercept, 'b')
plt.show()
```

<br>

# 🌷 iris dataset으로 회귀분석 : `ols()` 

`ols` 상관관계의 강도에 따라 회귀분석 모델의 신뢰성을 판단한다.  


```py
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris.head(3))

sns.pairplot(iris, hue='species', height=1.5)
plt.show()
```

<br>

## 상관관계 확인  

```py
print(iris.corr())
```

<br>

## 상관관계가 약한 변수로 모델 작성  


```py
result1 = smf.ols(formula='sepal_length ~ sepal_width', data = iris).fit()
print(result1.summary())

print('\n설명력:', result1.rsquared)
print('p-value:', result1.pvalues)
```

결정계수가 R-squared: 0.014 로 설명력이 매우 약하다.  
또한 독립변수 sepal_width에 대한 p-value가 0.152로 0.05보다 크므로 종속변수에 영향을 준다고 보기 어려워 독립변수에 부적합하다.  
그러므로 이 모델은 쓸만한 모델이라고 할 수 없다.

<br>

## 상관관계가 강한 변수로 모델 작성

```py
result2 = smf.ols(formula='sepal_length ~ petal_length', data = iris).fit()
print(result2.summary())

print('\n설명력:', result2.rsquared)
print('p-value:', result2.pvalues)
```

결정계수로 R-squared: 0.760 설명력이 높으며  
p-value가 1.038667e-47 < 0.05 이므로 petal_length는 독립변수로 적합하다.  
또한 이 모델은 좋은 모델이라고 할 수 있다.  


즉, 변수 간 상관관계가 강해야 좋은 모델이 만들어질 수 있다.

<br>

## 예측값 확인


```py
#기존데이터
print(iris['petal_length'][:5])
print('실제값:', iris['sepal_length'][:5])
print('예측값:', result2.predict(iris['petal_length'][:5]))

#새로운 데이터
new_data = pd.DataFrame({'petal_length':[1.0,1.8,3.8]})
y_pred = result2.predict(new_data)
print('예측값:\n', y_pred)
```

<br>

## 참고 : 수식만들기  

R에서의 formula = 'sepal_length ~.' 방법은 사용 불가하다. 많은 독립변수를 한번에 입력하는 방법을 알아보자  


```py
column_select = ' + '.join(iris.columns.difference(['sepal_length','species']))
print(column_select)

my_formula = 'sepal_length ~ ' + column_select
print(my_formula)

result3 = smf.ols(formula = my_formula , data = iris).fit()
print(result3.summary())
```

<br>

# **다중선형회귀 (Multiple Linear Regression)**  

회귀분석에서 독립변수가 2개 이상일 경우 다중회귀분석이라고 한다.   

```py
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/content/drive/MyDrive/testdata/drinking_water.csv')

model2 = smf.ols('만족도 ~ 적절성 + 친밀도', data=df).fit()
print(model2.summary())
```

<br>

# 🚗 mtcars dataset으로 선형회귀 분석 : `ols()`

```py
import statsmodels.api
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(mtcars.columns)
print()

print(mtcars.describe())
print(mtcars.info())

print(np.corrcoef(mtcars.hp, mtcars.mpg))  #마력수, 연비
print(np.corrcoef(mtcars.wt, mtcars.mpg))  #차체무게, 연비
```
<br>

## 시각화 

```py
plt.scatter(mtcars.hp, mtcars.mpg)
plt.xlabel('마력수')
plt.ylabel('연비')

#추세선그리기
slope, intercept = np.polyfit(mtcars.hp, mtcars.mpg, 1)
plt.plot(mtcars.hp, mtcars.hp * slope + intercept, 'r')
plt.show()
```

<br>

## 단순선형회귀 모델  


```py
result = smf.ols('mpg ~ hp', data=mtcars).fit()
print(result.summary())
print()

#신뢰구간
print(result.conf_int())
print(result.conf_int(alpha=0.05))
print(result.summary().tables[1])

print(-0.0682 * 110 + 30.0989)
print(-0.0682 * 50 + 30.0989)
print(-0.0682 * 250 + 30.0989)
```

설명력이 0.602이고
hp에 대한 p-value가 0.000 > 0.05 이므로 hp는 적합한 독립변수라고 할 수 있다.  


```py
#추정치 구하기 - 차체 무게를 입력해 연비를 추정
result3 = smf.ols('mpg ~ wt', data=mtcars).fit()
print(result3.summary().tables[1])
print('결정계수:', result3.rsquared)

pred = result3.predict()
#print(pred)
print(mtcars.mpg[0])  #실제값
print(pred[0])  #예측값
print()

#DataFrame으로 보기
data = {
    'mpg':mtcars.mpg,
    'mpg_pred':pred
}
df = pd.DataFrame(data)
print(df)
```
<br>

```py
#새로운 차체 무게에 대한 연비 추정

mtcars.wt = float(input('차체 무게 입력:'))
new_pred = result3.predict(pd.DataFrame(mtcars.wt))

print('차체무게: {}일 때 예상연비는 {}'.format(mtcars.wt[0], new_pred[0]))

new_wts = pd.DataFrame({'wt':[6,3,0.5]})
new_pred2 = result3.predict(new_wts)

print('예상연비:\n', new_pred2)
print('\n예상연비:\n', np.round(new_pred2.values, 2))
```

<br>

## 다중회귀 모델 

```py
result2 = smf.ols('mpg ~ hp + wt', data=mtcars).fit()
print(result2.summary())
print()

print(result.conf_int(alpha=0.05))
print((-0.0318 * 110) + (-3.8778 * 5) + 37.2273)
```

<br>

# Kaggle의 Advertising dataset : `ols()`  


```py
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

adv = pd.read_csv('/content/drive/MyDrive/testdata/Advertising.csv', usecols=[1,2,3,4])
print(adv.head(5))

print(adv.index, adv.columns)
print(adv.info())
print()

#상관계수 확인
print(adv.corr())
print(adv.loc[:,['sales','tv']].corr())
print()

#모델 작성
#lm = smf.ols(formula='sales ~ tv', data=adv)
#lm_learn = lm.fit()

lm = smf.ols(formula='sales ~ tv', data=adv).fit()
print(lm.summary())
print()

print(lm.params)
print(lm.pvalues)
print(lm.rsquared)
```

<br>

## 시각화


```py
import seaborn as sns

plt.scatter(adv.tv, adv.sales)
plt.xlabel('tv')
plt.ylabel('sales')

x = pd.DataFrame({'tv':[adv.tv.min(), adv.tv.max()]})
y_pred = lm.predict(x)
plt.plot(x, y_pred, c='red')

plt.title('Linear Regression')
plt.xlim(-50, 400)
plt.ylim(ymin=0)
plt.show()
```

<br>

## 예측 1 : 새로운 tv 값으로 sales 추정 : 단순회귀  


```py
x_new = pd.DataFrame({'tv':[500, 50, 1000]})
pred = lm.predict(x_new)

print('예상 매출:\n', pred.values)
```

<br>

## 예측 2 : 다중회귀

### 모델작성

```py
#newspaper 포함
lm = smf.ols(formula='sales ~ tv + radio + newspaper', data=adv).fit()
print(lm.summary())

#newspaper 제외
lm_mul = smf.ols(formula='sales ~ tv + radio', data=adv).fit()
print(lm_mul.summary())
```

자기상관계수 Durbin-Watson는 모두 2점대로 안정적  

newspaper 변수 포함 시 Adj. R-squared: 0.896  
newspaper 변수 제외 시 Adj. R-squared: 0.896  

newspaper 변수의 p-value 0.860 > 0.05  

상관관계가 낮았던 변수인 newspaper의 제외 전, 후 모델의 결정계수에 차이가 없다.  
즉, newspaper 변수는 모델의 설명력에 영향을 끼치지 않는다. 제외하고 모델을 만들어도 무방하다.

<br>

새로운 tv, radio 값으로 sales 예측하기

```py
x_new2 = pd.DataFrame({'tv':[200, 55.5, 100], 'radio':[30.1, 45.5, 50.1]})
pred2 = lm_mul.predict(x_new2)

print('추정값:', pred2.values)
```

<br>

# 🌞 회귀모델의 적절성을 판단하는 기준  

>1. 정규성 : 독립변수들의 잔차항이 정규분포를 따라야 한다.    
2. 독립성 : 독립변수들 간의 값이 서로 관련성이 없어야 한다. 
3. 선형성 : 독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지면 좋지 않다.  
4. 등분산성 : 독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.  
5. 다중공선성 : 독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.  


회귀분석에서 잔차는 정규성, 등분산성, 독립성을 가지는 것으로 가정한다.  


상관관계가 높은 변수들이 여러 개 존재하면 문제가 발생한다.    
표준오차가 비정상적으로 커지게 되는데 그러면 t-value 값이 작아지게 되어 반비례 관계인 p-value 값이 커지기 때문이다.  
p-value가 커지면 독립변수가 유의하지 않게 된다.


등분산성과 다중공선성은 독립변수가 2개 이상일때 적용된다.

<br>

## 잔차항 구하기  

= 실제값-예측값

```py
fitted = lm_mul.predict(adv)
residual = adv['sales'] - fitted
print('residual', residual)
```

<br>

## 1. 정규성 

독립변수들의 잔차항이 정규분포를 따라야 한다.

<br>

### Q-Q plot으로 확인

```py
import scipy.stats

sr = scipy.stats.zscore(residual)  #표준정규분포 표준화
(x, y), _ = scipy.stats.probplot(sr)
sns.scatterplot(x, y)

plt.plot([-3, 3], [-3, 3], '--', color='gray')
plt.show()
```

<br>

### shapiro wilk test  


```py
print('shapiro test:', scipy.stats.shapiro(residual))
```
p-valaue가 4.190036317908152e-09 < 0.05 이므로 정규성을 만족하지 못한다.

<br>

## 2. 독립성  

독립변수들 간의 값이 서로 관련성이 없어야 한다.   
잔차가 자기상관(인접 관측치의 오차 상관여부)이 있는 지 확인

<br>

### Durbin-Watson test : `model_name.summary()`

```py
print(lm_mul.summary())
```
0~4 사이의 값을 가지는데 2에 가까울 수록 자기상관이 없다.  

Durbin-Watson: 2.081 로 자기상관이 없다. 독립적이다.  
잔차항이 독립성을 만족한다.

<br>

## 3. 선형성 

독립변수의 변화에 따라 종속변수도 변화하나 일정한 패턴을 가지는 것은 좋지 않다.

<br>

### 그래프로 확인


```py
import seaborn as sns

sns.regplot(fitted, residual, lowess=True, line_kws={'color':'red'})  
plt.plot([fitted.min(), fitted.max()], [0, 0], '--', color='gray')
plt.show()
```
회색선에 가까울 수록 선형성을 만족한다고 할 수 있다. 선형성을 완전히 만족한다고 보기는 어렵다.  


아웃라이어(이상치) 등을 제거하거나, 로그를 씌우는 등의 방법으로 해결할 수 있다.

<br>

## 4. 등분산성  

독립변수들의 오차(잔차)의 분산은 일정해야 한다. 특정한 패턴 없이 고르게 분포되어야 한다.

<br>

### 그래프로 확인

```py
sr = scipy.stats.zscore(residual)

sns.regplot(fitted, np.sqrt(np.abs(sr)), lowess=True, line_kws={'color':'red'})
```
일정하지 않고 약간의 패턴이 보인다.  
등분산성을 만족하지 못한다고 볼 수 있다.  

이상치 확인, 비선형 관계 확인, 정규성 확인, 가중회귀분석 등을 고려해보도록 한다.  

가중회귀분석 : 가중 회귀 분석은 잔차의 분산이 일정하다는 최소 제곱법 가정이 어긋나는 경우(이분산성) 사용할 수 있는 방법. 
올바른 가중치를 사용하여 가중된 잔차 제곱합을 최소화함으로써 분산이 일정한(동분산성) 잔차를 만들어낸다.  


<br>

## 5. 다중공선성 (multicollinearity) : VIF  


독립변수들 간에 강한 상관관계로 인한 문제가 발생하지 않아야 한다.  

VIF(Variance Inflation Factors, 분산팽창요인)을 통해 확인한다.  
VIF가 10을 넘으면 해당 변수로 인해 다중공선성이 발생할 수 있다고 판단하고 5 이상이면 주의할 필요가 있다고 할 수 있다. 

다중공선성이 발생하면 해당 변수를 제거하는 방법으로 해결할 수 있다. 그러나 변수를 제거하는 것은 중요한 문제이므로 신중하게 선택해야 한다.

```py
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

adv = pd.read_csv('/content/drive/MyDrive/testdata/Advertising.csv', usecols=[1,2,3,4])

#컬럼별로 확인
print(variance_inflation_factor(adv.values, 0))
print(variance_inflation_factor(adv.values, 1))
print(variance_inflation_factor(adv.values, 2))
print(variance_inflation_factor(adv.values, 3))
print()

#데이터프레임으로 저장
vif = pd.DataFrame()
vif['vif_value'] = [variance_inflation_factor(adv.values, i) for i in range(adv.shape[1])]
print(vif)
```

<br>

## 이상치(극단값) 확인 : Cook's Distance 


```py
from statsmodels.stats.outliers_influence import OLSInfluence

cd, _ = OLSInfluence(lm_mul).cooks_distance
print(cd.sort_values(ascending=False).head(9))  #출력된 값들이 outlier
```

<br>

그래프


```py
import statsmodels.api as sm

sm.graphics.influence_plot(lm_mul, criterion='cooks')
plt.show()

print(adv.iloc[[130, 5, 35, 178, 126]])
```
이상치 중에서도 중요한 역할을 하는 값일 수도 있다.  
그러므로 이상치라고 모두 삭제하는 것이 아니라 데이터를 확인해가며 신중하게 삭제하도록 한다.

<br>

# 👉 `LinearRegression()`

<br>

## hp(마력)가 mpg(연비)에 미치는 영향

```py
import statsmodels.api
from sklearn.linear_model import LinearRegression

mtcars = statsmodels.api.datasets.get_rdataset('mtcars').data
print(mtcars.head(3))
print(type(mtcars))

x = mtcars[['hp']].values  #feature는 반드시 matrix형태로 입력
y = mtcars['mpg'].values   #label은 matrix or vector
#y = mtcars[['mpg']].values
print(x[:3], type(x))
print(y[:3], type(y))

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.show()
```

모델 학습
```py
fit_model = LinearRegression().fit(x, y)
print('기울기(회귀계수, w):', fit_model.coef_[0])  
print('절편(편향, b):', fit_model.intercept_)
```

예측
```py
pred = fit_model.predict(x)  #학습된 data로 전체 자료에 대한 예측 결과를  사실 예측이라기보다는 모델 평가
#print(pred)
print('예측값:', pred[:5])
print('실제값:', y[:5])
```

<br>

### RMSE(평균제곱근 오차)로 모델의 성능 평가  

평균 제곱근 오차(Root Mean Square Error; RMSE)는 추정 값 또는 모델이 예측한 값과 실제 환경에서 관찰되는 값의 차이를 다룰 때 흔히 사용한다.  정밀도(precision)를 표현하는데 적합하다.  

각각의 차이값은 잔차(residual)라고도 하며, 평균 제곱근 오차(편차)는 잔차들을 하나의 측도로 종합할 때 사용된다.  

```py
import numpy as np
from sklearn.metrics import mean_squared_error

lin_mse = mean_squared_error(y, pred)
lin_rmse = np.sqrt(lin_mse)
print('RMSE:', lin_rmse)
```
<br>

### 새로운 hp에 대한 mpg 

```py
new_hp = [[110]]
new_pred = fit_model.predict(new_hp)
print('%s 마력인 경우 연비는 %s'%(new_hp[0][0], new_pred[0]))

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
print(type(iris))
print(iris.data[:3])  #배열 형태
print(iris.feature_names)
print(iris.target[75:80])
print(iris.target_names)
print()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['target_names'] = iris.target_names[iris.target]
print(iris_df[:5])
```

<br>

## 과적합 방지를 위해 데이터셋 나누기 : `train_test_split()`

시계열 데이터의 경우 데이터를 랜덤하게 추출하면 안됨. shuffle X. `shuffle=False` 설정한다.  

```py
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(iris_df, test_size = 0.3)
print(iris_df.shape)
print(train_set.shape)
print(test_set.shape)
#랜덤하게 분리됨
```

<br>

### 분석 모델 작성 1

```py
from sklearn.linear_model import LinearRegression

#상관관계 확인
pd.set_option('max_columns', None)
#print(iris_df.corr())

model_lr = LinearRegression().fit(X = train_set.iloc[:,[2]], y = train_set.iloc[:,[3]])
print(model_lr.coef_)
print(model_lr.intercept_)
print()

print('LR predict:\n', model_lr.predict(test_set.iloc[:,[2]][:5]))
print('real data:\n', test_set.iloc[:,3][:5])
```

<br>

### 시각화  


```py
import matplotlib.pyplot as plt

plt.scatter(train_set.iloc[:,[2]], train_set.iloc[:,[3]], c='black')
plt.plot(test_set.iloc[:,[2]], model_lr.predict(test_set.iloc[:,[2]]))
plt.show()
```
<br>


references: 
https://pro-jm.tistory.com/37  
https://support.minitab.com/ko-kr/minitab/18/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/weighted-regression/


<br>

