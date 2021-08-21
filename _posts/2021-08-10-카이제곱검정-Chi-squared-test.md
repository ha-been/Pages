<br>

# ⭐ SciPy를 이용한 확률 분포 분석

<br>

## 확률분포 객체 생성  


객체생성 함수  

|종류|	이름|	확률 분포|
|----|------|------|
|이산|	bernoulli|	베르누이 분포|
|이산|	binom|	이항 분포|
|연속|	uniform|	균일 분포|
|연속|	norm|	가우시안 정규 분포|
|연속|	beta|	베타 분포|
|연속|	gamma|	감마 분포|
|연속|	t|	스튜던트 t 분포|
|연속|	chi2|	카이 제곱 분포|
|연속|	f|	F 분포|
|연속|	dirichlet|	디리클리 분포|
|연속|	multivariate_normal|	다변수 가우시안 정규 분포|

<br>

## 정규분포 그리기 : `stats.norm()`

```py
import numpy as np
from scipy import stats

np.random.seed(1)
print(stats.norm(loc=1, scale=2).rvs(10))  #정규분포 기댓값: 1, 표준 편차: 2, rvs(): 랜덤샘플생성
```

분포의 모수(parameter)를 인수로 지정해 주어야 한다.  
분산, 표준편차의 중요성 : 분포 파악이 가능하다. 


|모수(인수)| 의미|
|----|----|
|`loc`|	일반적으로 분포의 기댓값|
|`scale`|	일반적으로 분포의 표준편차|
|`size`|	샘플 생성시 생성될 샘플의 크기|
|`random_state`|	샘플 생성시 사용되는 시드(seed)값|

<br>

|메서드|기능|
|----|----|
|`pdf`|확률 밀도 함수 (probability density function)  |
|`pmf`|확률 질량 함수 (probability mass function)  |
|`cdf`|누적 분포 함수 (cumulative distribution function)   |
|`rvs`|랜덤 샘플 생성 (random variable sampling)  |
|`stats`|기술 통계 함수 (descriptive statistics)  |
|`fit`|모수 추정 (parameter estimation)|

<br>

## 그래프로 확인

```py
import matplotlib.pyplot as plt

centers = [1, 1.5, 2]
col = 'rgb'

std = 0.1
data_1 = []
for i in range(3):
  data_1.append(stats.norm(centers[i], scale=std).rvs(100))
  #print(data_1)

  plt.plot(np.arange(len(data_1[i])) + i * len(data_1[0]), data_1[i], '*', c=col[i])

plt.show()
```

<br>

# **Chi Squared Test**

```py
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

<br>

# 📝 **카이제곱검정**  

카이제곱 검정(chi-squared test) 또는 χ² 검정은 카이제곱 분포에 기초한 통계적 방법으로,  관찰된 빈도(관측빈도)가 기대되는 빈도(예측빈도)와 의미있게 다른 지 즉, 차이가 통계적으로 유의한지 여부를 검정하기 위해 사용되는 방법이다. 자료가 빈도로 주어졌을 때, 특히 명목척도(범주형) 자료의 분석에 이용되며 확률모형이 전반적으로 자료를 얼마나 잘 설명하는지 검정한다.  

집단 간의 동질성 여부, 두 변인 간의 상관성을 통계적으로 검증하고자 할 때 사용한다.  
수집된 자료가 모집단에 대한 분포형태를 가정할 수 없는 경우에 사용하는 비모수적 통계방법의 일종이다.  
단, 두 변인(범주) 간의 인과관계는 파악할 수 없다.  

빈도표를 작성하고 이를 바탕으로 분석한다.  

<br>

1. 적합성 검정   
범주형 자료에 대해 관측된 값이 기대한 값과 얼마나 차이가 나는 지, 관측치가 특정 분포를 따르는 지를 검정한다.  

2. 동일성 검정  
두 집단의 분포가 동일한지를 검정한다.  

3. 독립성 검정  
모집단에서 추출한 자료들이 두 가지 변수 A, B에 의해서 범주화되어 있을 때 (분할표 상의 변수), 이 두 변수 A와 B 사이에 연관성이 있는 지를 검정하는 것이다.  
즉, 두 변수 사의의 독립성을 검정한다.

<br>

## 🔹 카이제곱 통계량 : χ²

카이제곱 통계량은 데이터의 분포와 가정된 분포 사이의 차이를 나타내는 측정값이다.  
카이제곱 통계량이 카이제곱 분포를 따른다면 카이제곱 분포를 사용해 가설검정을 수행한다. 

> 카이제곱 값 χ² = ∑(관측값 - 기대값)² / 기대값

<br>

## 🔹 P-Value

카이제곱 검정을 통해 P-Value를 얻게 된다. P-Value는 검정 결과가 얼마나 유의한 지를 나타내는 지표로 사용된다.  

P-Value는 검정 통계량에 관한 확률로 검정 통계량보다 크거나 같은 값을 얻을 수 있을 확률을 의미한다.  
검정 통계량은 거의 대부분이 귀무가설을 가정하고 얻게 되는 값이다.  
즉, 귀무가설을 가정한 분포와 동일할 확률이다.  

그러므로 P-Value가 유의수준보다 작으면 귀무가설을 기각하고 대립가설을 채택하게 된다.

<br>

카이제곱 검정을 수행하고, P-Value를 얻기 위해서는 두 가지의 정보가 필요하다.  

> 1) 자유도  
2) 유의수준(α) : 연구자에 의해 결정됨

<br>

## 🔹 자유도 (Degree of Freedom)

통계적 추정을 할 때 표본자료 중 모집단에 대한 정보를 주는 독립적인 자료의 수를 말한다.  

자유도는 모분산을 모르기 때문에 필요하다.  
표본의 분산은 모집단의 분산보다 항상 작아지는 경향을 보이기 때문에 (편향이 발생) 표본의 분산을 모집단의 분산에 근사하게 하기 위하여 일정 비율을 곱해준다.  
그 비율은 n / (n-1) 이며, 표본의 분산에 이 비율을 곱하면 모집단의 분산에 근사하게 된다.  

또한 분산의 원래 계산식에 있는 분모의 n이 약분되기 때문에 (n-1)만 남게된다.  
즉, 표본의 분산을 구할 때, n 대신 n-1을 나누면 표본의 분산을 모집단의 분산에 근사해지게 할 수 있는 것이다.   

따라서 자유도는 표본의 평균을 구할 때는 사용되지 않고 표본의 분산을 구할 때만 사용된다.  


> df = N - K  혹은 N-1  
자유도 = 사례수 - 통계적제한조건수  (범주의수-1)

카이제곱에서의 자유도  
> r행 c열의 분할표에서 카이제곱 통계량의 자유도는 (r-1)*(c-1) 이다.

<br>

## 예제 1

> 가설설정  
귀무가설 : 벼락치기로 공부하는 것과 합격 여부는 관계가 없다.  
대립가설 : 벼락치기로 공부하는 것과 합격 여부는 관계가 있다.

<br>

### 데이터 준비

벼락치기 공부 여부에 따른 합격, 불합격 데이터

```py
import pandas as pd
data = pd.read_csv('/content/drive/MyDrive/testdata/pass_cross.csv', encoding = 'euc-kr')
print(data.head(3))
print(data.shape[0], data.shape[1])

print(data[(data['공부함']==1) & (data['합격']==1)].shape[0])
print(data[(data['공부함']==1) & (data['불합격']==1)].shape[0])
```

<br>

빈도표 작성

```py
data2 = pd.crosstab(index = data['공부안함'], columns = data['불합격'], margins = True)
data2.columns = ['합격','불합격','행합']
data2.index = ['공부함','공부안함','열합']
print(data2)
```

<br>

### 👉 수식을 이용해 χ² 값 구하기


기대값 구하기 : (각 행의 주변값) * (각 열의 주변값) / 총합

```py
chi2 = (18 - 15)**2 / 15 + (7 - 10)**2 / 10 + (12 -15)**2 / 15 + (13 - 10)**2 / 10
print(chi2)
```

자유도는 (2-1)*(2-1) ==> 1 

임계값은 카이제곱 분포표 사용 : 3.84  

결론 : chi2값 3.0은 임계값 3.84보다 작으므로 귀무채택역에 chi2값이 존재하여 귀무가설을 채택한다.   
즉, 벼락치기 공부를 하는 것과 합격여부는 관계가 없다고 할 수 있다.

<br>

### 👉 모듈이 지원하는 함수를 사용해 χ²값과 p-value 구하기 : `chi2_contingency()`  

χ²값과 p-value는 반비례 관계이다.

```py
import scipy.stats as stat

chi2, p, ddof, excepted = stats.chi2_contingency(data2)
print('chi2 :', chi2)
print('p :', p)                #p-value
print()
print('ddof :', ddof)
print('excepted :', excepted)  #예측값
```

유의수준(α=0.05) < p-value(0.557825400) 이므로 귀무가설을 채택한다.

<br>

# 📝 **일원카이제곱검정 (One-way χ² test)**

적합도 (goodness of fit) 검정이라고도 한다.  한 개의 범주형 요인 (하나의 변수) 을 사용.  
관측값들이 어떠한 이론적 분포를 따르고 있는지 확인한다.  

ex) 꽃 색깔의 표현 분리비율이 3:1이 맞는가?

<br>

## 🔹 적합도 검정  : `chisquare()`

주사위를 60 회 던져서 나온 관측도수 / 기대도수가 아래와 같이 나온 경우에 이 주사위는 적합한 주사위가 맞는가를 일원카이제곱 검정
으로 분석

> 주사위눈금   1  2  3  4  5  6  
관측도수       4  6 17  16 8  9  
기대도수       10 10 10 10 10 10  

<br>

가설설정

> 귀무가설 : 주사위는 게임에 적합하다.  -> 기대치와 관측치는 차이가 없다. 독립적이다. 상관이 없다.  
대립가설 : 주사위는 게임에 적합하지 않다. -> 기대치와 관측치는 차이가 있다. 독립적이지 않다. 상관이 있다.

```py
import pandas as pd
import scipy.stats as stats

data = [4, 6, 17, 16, 8, 9]
data2 = [10, 10, 10, 10, 10, 10]
print(stats.chisquare(data))
print()

result1 = stats.chisquare(data)
print('통계량(chi2) : %.5f, p-value: %.5f'%result1)
print()

result2 = stats.chisquare(data, data2)
print('통계량(chi2) : %.5f, p-value: %.5f'%result2)
```

해석1:  
p-value:0.01439 < 0.05 이므로 유의미한 수준(α=0.05)에서 귀무가설을 기각하고, 대립가설을 채택한다.  
즉, 현재 실험에 사용된 주사위는 게임에 적합하지 않다.  

해석2:  
카이제곱분포를 사용한 결과 임계값 : 11.07 < 14.2000 이므로 귀무가설을 기각하고 대립가설을 채택한다.

<br>

## 🔹 선호도 검정  

5개의 스포츠 음료에 대한 선호도에 차이가 있는지 검정하기 : drinkdata.csv

```py
data3 = pd.read_csv('/content/drive/MyDrive/testdata/drinkdata.csv')
print(data3)
print()

print(stats.chisquare(data3['관측도수']))
```

해석:  
p-value=0.00039991 < 0.05 이므로 귀무가설을 기각하고 대립가설을 채택한다.

<br>

# 📝 **이원카이제곱검정  (Two-way χ² test)**

두 개 이상의 집단 또는 범주의 변인을 대상으로 독립성, 동질성 검정
- 동일 집단의 두 변인(ex 학력수준과 대학진학 여부)을 대상으로 관련성이 있는지 없는지 확인  
- 독립성 검정은 두 변수 사이의 연관성으로 검정한다.

<br>

## 🔹 독립성 검정

실습 : 교육수준과 흡연 간의 관련성 분석 : smoke.csv

<br>

가설설정

> 귀무가설 : 교육수준과 흡연은 관련이 없다. (독립이다.)  
대립가설 : 교육수준과 흡연은 관련이 있다. (독립이 아니다.)

```py
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('/content/drive/MyDrive/testdata/smoke.csv')
print(data.head(3))
print(data.tail(3))
print()

print(data['education'].unique()) #대학원졸 대졸 고졸
print(data['smoking'].unique())   #과흡연 보통 노담
```

<br>

### 교차표 작성 : `crosstab()`

교육수준별 흡연 인원수를 파악하기 위해 교차표 작성한다. 

야트보정 : 분할표의 자유도가 1인 경우는 χ²값이 약간 높게 계산된다.  
그래서 절대값 |O-E|에서 0.5를 뺀 다음 제곱하여 적용한다.  
이를 야트보정이라고 하며 chi2_contingency에서는 보정이 자동으로 이루어진다

```py
ctab = pd.crosstab(index = data['education'], columns = data['smoking'])  #normalize=True : 비율로 출력. 현재는 범주형데이터라 적용X
ctab.index = ['대학원','대학','고등교육']
ctab.columns = ['과흡연','흡연','비흡연']

print(ctab)
```

<br>

### Scipy의 `chi2_contingency()`


```py
chi_result = [ctab.loc['대학원'], ctab.loc['대학'], ctab.loc['고등교육']]
chi2, p, ddof, exp = stats.chi2_contingency(chi_result)
#chi2, p, ddof, exp = stats.chi2_contingency(ctab)   #현재 데이터는 가능

print('chi2 :', chi2)
print('p :', p)
```

해석:  
p-value (0.00081825) < 유의수준 (0.05) 이므로 귀무가설을 기각하고 대립가설을 채택한다.

<br>

### 예제 1) 독립성 검정 : 인종 간 인원수 👧🏿👩👱🏻‍♂️

국가 전체와 지역에 대한 인종 간 인원수로 독립성 검정을 실습해보자.  

두 집단 (국가 전체 - national, 특정지역 - la) 의 인종간 인원수의 분포가 관련이 있는가?

```py
import pandas as pd
import scipy.stats as stats

national = pd.DataFrame(['white']*100000 + ['hispanic']*60000 + ['black']*50000 + ['asian']*15000 + 
                        ['other']*35000)
la = pd.DataFrame(['white']*600 + ['hispanic']*300 + ['black']*250 + ['asian']*75 + 
                        ['other']*150)
print(national)
print()
print(la)

na_table = pd.crosstab(index=national[0], columns='count')
la_table = pd.crosstab(index=la[0], columns='count')
print(na_table)
na_table['count_la'] = la_table['count']
print(na_table)

chi2, p, ddof, _ = stats.chi2_contingency(na_table)
print("chi2 : ", chi2)# chi2 :  18.099524243141698
print("p : ", p)# p :  0.0011800326671747886
print("ddof : ", ddof)# ddof :  4
```

해석 : p-value 0.0011800326671747886 < 0.05 유의수준 보다 매우작기 때무에 귀무가설은 기각하고 대립가설을 채택한다.

<br>

## 🔹 동질성 검정

두 집단의 분포가 동일한가 다른 가를 검증하는 방법이다. 두 집단 이상에서 각 범주(집단) 간의 비율이 서로 동일한가를 검정하게 된다.  
두 개 이상의 범주형 자료가 동일한 분포를 갖는 모집단에서 추출된 것인지 검정하는 방법이다.

```py
import pandas as pd
import scipy.stats as stats

#동질성 검정실습 1) 교육방법에 따른 교육생들의 만족도 분석 - 동질성 검정 survey_method.csv
data = pd.read_csv('/content/drive/MyDrive/testdata/survey_method.csv')
print(data.head(5))
print()

print(data['method'].unique())

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족','만족','보통','불만족','매우불만족']
ctab.index = ['방법1','방법2','방법3']

print(ctab)

chi2_result = [ctab.loc['방법1'], ctab.loc['방법2'], ctab.loc['방법3']]
chi2, p, ddof, _ = stats.chi2_contingency(chi2_result)
msg = '검정통계량 chi2: {}, p-value: {}, ddof: {}'
print(msg.format(chi2, p, ddof))
```

해석:  
p값 0.5864 > 0.05 이므로 귀무가설을 채택한다.

교육방법에 따른 교육생들의 만족도에 차이가 없다.

<br>

### 예제 2) 동질성 검정 : 연령대별 SNS 이용률 📳

연령대별 SNS 이용률의 동질성 검정  
20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS 서비스들에 대해 이용 현황을 조사한 자료를 바탕으로  연령대별로 홍보전략을 세우고자 한다.  연령대별로 이용현황이 서로 동일한지 검정해보도록 하자.  

<br>

가설설정  
> 귀무가설 : 연령대별로 SNS 서비스 이용현황이 동일하다.  
대립가설 : 연령대별로 SNS 서비스 이용현황이 동일하지 않다.

```py
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('/content/drive/MyDrive/testdata/snsbyage.csv')
print(data.head(5))
print(len(data))
print()

print(data['age'].unique())
print(data['service'].unique())
```

빈도표

```py
ctab = pd.crosstab(index=data['age'], columns=data['service'])  #나이별 서비스 이용 인원수
print(ctab)

chi2, p, df, exp = stats.chi2_contingency(ctab)
msg = 'chi2: {}, p-value: {}, ddof: {},'
print(msg.format(chi2, p, df))
```

p-value가 1.1679064204212775e-18으로 유의수준 0.05보다 작기 때문에 귀무가설을 기각하고 대립가설을 채택한다.  

연령대별로 SNS 서비스 이용현황이 동일한지 1439명을 대상으로 설문조사한 결과,  
연령대에 따라 주로 사용하는 SNS 서비스는 서로 다른 것으로 볼 수 있다.

<br>

### 샘플링  

위 데이터는 표본이지만 모집단이라 가정하고 샘플링을 진행해본다.

```py
sample_data = data.sample(n=500, replace=True)  #복원추출
print(sample_data.head(5), len(sample_data))

ctab2 = pd.crosstab(index=sample_data['age'], columns=sample_data['service'])  
print(ctab2)

chi2, p, df, exp = stats.chi2_contingency(ctab2)

msg = 'chi2: {}, p-value: {}, ddof: {},'
print(msg.format(chi2, p, df))
```

샘플링을 통해서도 귀무가설을 기각하는 결과가 나오는 것을 확인할 수 있다.

<br>

references:  
https://namyoungkim.github.io/scipy/probability/2017/09/04/scipy/  
https://bioinformaticsandme.tistory.com/139  
https://data-gardner.tistory.com/23
"""

<br><br>
