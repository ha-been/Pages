
<br>

# 📝 데이터의 시각화 : `Matplotlib`, `Seaborn`

`Matplotlib`: `pyplot` 모듈의 함수를 사용해서 그래프를 만들어 시각화를 할 수 있는 패키지이다.  
`Seaborn`: `Matplotlib`를 보완하기 위한 라이브러리로, Matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지이다.  

> 시각화를 해야 하는 이유
1. 많은 양의 데이터를 한 눈에 볼 수 있다.  
2. 누구나 쉽게 인사이트를 찾을 수 있다.  
3. 보다 정확하게 데이터를 이해할 수 있다.  
4. 다른 사람에게 데이터 인사이트를 공유하는 데에 효과적이다.  
5. 데이터가 존재하는 다양한 분야에서 활용 가능하다.  


<br>


# 📊 **Matplotlib**

```py
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline b   #매직 명령어(매직 커맨드) jupyter에서 실행 시
```

<br>

## 👉 선 그래프  

시간, 순서, 크기 등을 표현하기에 적절한 그래프  

```py
plt.rc('font', family='NanumBarunGothic')   #한글 깨짐 방지  
plt.rcParams['axes.unicode_minus'] = False  #음수 깨짐 방지

x = ['서울', '인천', '수원']  #list, tuple 가능. setX
y = [5, 3, 7]

plt.xlim([-1, 3])
plt.ylim([0, 10])
plt.yticks(list(range(0, 11, 3)))  #y축 변경

plt.plot(x, y)
plt.show()      #코랩에서는 안해줘도 무방함

data = np.arange(1, 11, 2)
print(data)

x = [0, 1, 2, 3, 4]
for a, b in zip(x, data):
  plt.text(a, b, str(b))
plt.plot(data)

plt.plot(data, data, 'r')
for a, b in zip(data, data):
  plt.text(a, b, str(b))
plt.show()

x = np.arange(10)
y = np.sin(x)
print(x,y )

#plt.plot(x,y)
#plt.plot(x, y, 'go-')
#plt.plot(x, y, 'go--')  #dash
#plt.plot(x, y, 'go:')
plt.plot(x, y, c='g', marker='o', ms=15, ls='--', lw=5)  #ms: marker size ls: line style

plt.show()
```


<br>

### 🌼 Hold : 그래프 겹쳐그리기

하나의 Figure 내에 복수 개의 Plot 그리기


```py
x = np.arange(0, np.pi*3, 0.1)
print(x)
print()

y_sin = np.sin(x)
y_cos = np.cos(x)

#너비와 높이 조정
plt.figure(figsize=(10, 5))
plt.plot(x, y_sin, 'r')
plt.scatter(x, y_cos) 

#축이름 지정
plt.xlabel('x축')
plt.ylabel('y축')
plt.title('sin, cos graph')

#범례
plt.legend(['sin','cos'])
plt.show()
```

<br>

### 🌼 Subplot : Figure를 여러 개로 분할


```py
x = np.arange(0, np.pi*3, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.subplot(2, 1, 1)  #행개수, 열개수, 활성화된 위치.  2행 1열 중 1행
plt.plot(x, y_sin)
plt.title('Sine')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')
plt.show()
```
<br>

```py
irum = ['a', 'b', 'c', 'd', 'e']
kor = [80, 50, 70, 70, 90]
eng = [60, 70, 80, 70, 60]

plt.plot(irum, kor, 'ro-')
plt.plot(irum, eng, 'bs-')

plt.ylim([0,100])
plt.legend(['국어','영어'], loc=4)  #loc : 범례의 위치

plt.grid(True)
plt.savefig('test1.png')   #차트를 이미지로 저장하기

#fig = plt.gcf()  
#fig.savefig('test.png')
```

<br>

### 저장된 그래프 불러오기  


```py
from matplotlib.pyplot import imread

img = imread('test1.png')
plt.imshow(img)
plt.show()
```

<br>

### 같은 방법으로 저장된 이미지 불러오기

```py
from matplotlib.pyplot import imread

img = imread('logo3.png')  #주의 jpg에서 png로 변환한 이미지의 경우 png로 로딩이 안됨
plt.imshow(img)
plt.show()
```

<br>

## 데이터에 맞는 그래프 선택이 매우 중요하다.

```py
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn

fig = plt.figure()  #명시적으로 차트영역 객체 선언

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.hist(randn(10), bins=10, alpha=0.8)  #bins:구간 수
ax2.plot(randn(10))
plt.show()
```

<br>

## 👉 막대그래프

```py
#세로막대그래프
data = [50, 80, 100, 70, 90]

plt.bar(range(len(data)), data)
plt.show()

#가로막대그래프
data = [50, 80, 100, 70, 90]

plt.barh(range(len(data)), data, alpha=0.5)
plt.show()
```

<br>

### 그래프에 오차 표시하기

```py
data = [50, 80, 100, 70, 90]
error = randn(len(data))

plt.barh(range(len(data)), data, alpha=0.5, xerr=error)
plt.show()
```
<br>

## 👉 다양한 그래프

<br>

#### Pie chart


```py
data = [50, 80, 100, 70, 90]

plt.pie(data, explode=(0,0.1,0,0.3,0), colors=['yellow','blue','red'])  #색상지정
plt.show()
```

<br>

#### Boxplot

```py
plt.boxplot(data)
````

<br>

#### Bubble chart

```py
n = 30
np.random.seed(0)
x = np.random.rand(n)
y = np.random.rand(n)

color = np.random.rand(n)
scale = np.pi*(15*np.random.rand(n))**2

plt.scatter(x, y, s=scale, c=color)
plt.show()
```

```py
import pandas as pd

fdata = pd.DataFrame(np.random.randn(1000, 4), index = pd.date_range('1/1/2000', periods=1000),
                     columns = list('abcd'))
fdata = fdata.cumsum()  #누적합
print(fdata)

plt.plot(fdata)
plt.show()
```

<br>

# 📈 **Seaborn**

## ✍ Titanic dataset으로 여러 그래프 그리기


```py
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset("titanic")
print(titanic.head(2))
print(titanic.info())
print(titanic[['sex','who']].tail(2))
print()

plt.hist(titanic['age'])
plt.show()
sns.displot(titanic['age'])
plt.show()

sns.boxplot(y='age', data=titanic, palette='Paired')
plt.show()

sns.relplot(x='who', y='age', data=titanic)
plt.show()

sns.countplot(x='class', data=titanic)
plt.show()

sns.countplot(x='class', data=titanic, hue='who')
plt.show()
```

<br>


## ✍ iris dataset 시각화


```py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/iris.csv')
print(iris_data.head(3))

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'])
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('iris data')
plt.show()

print(iris_data['Species'].unique())
print(set(iris_data['Species']))
print()

cols = []
for s in iris_data['Species']:  #종류별로 색상 지정하기 
  choice = 0
  if s == 'setosa': choice = 1
  if s == 'versicolor': choice = 2
  if s == 'virginica': choice = 3
  cols.append(choice)

plt.scatter(iris_data['Sepal.Length'], iris_data['Petal.Length'], c = cols)
plt.xlabel('Sepal.Length')
plt.ylabel('Petal.Length')
plt.title('iris data')
plt.show()
```


<br>

### Pandas의 Plotting 기능 사용

```py
iris_col = iris_data.loc[:, 'Sepal.Length':'Petal.Width']
#print(iris_col)

from pandas.plotting import scatter_matrix
scatter_matrix(iris_col, diagonal='kde')  #diagonal='kde': 밀도 추정 곡선 생성. hist, bar ...
#scatter_matrix(iris_col, diagonal='hist')
plt.show()
```

```py
import pandas as pd

np.random.seed(0)
df = pd.DataFrame(np.random.randn(10, 3), columns=['a', 'b', 'c'],
                  index = pd.date_range('1/1/2000', periods=10))
print(df)
print()

df.plot()
plt.show()

df.plot(kind = 'bar', rot=45)  #rot=45: label에 각도를 준다 
plt.xlabel('time')
plt.ylabel('data')
plt.show()
```

<br>

### Seaborn 사용

```py
import seaborn as sns

sns.pairplot(iris_data, hue='Species')

x = iris_data['Sepal.Length'].values
sns.rugplot(x)
plt.show()

sns.kdeplot(x)
plt.show()
```

<br>

# 🛴 자전거 대여 정보를 이용해 시각화  

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='NanumBarunGothic') 
plt.rcParams['axes.unicode_minus'] = False
#plt.style.use('ggplot')

train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/data/train.csv', parse_dates=["datetime"])
print(train.head(2), ' ', train.shape)
print(train.columns)
#print(train.info())
print(train.temp.describe())
```

<br>

## Null인 자료를 시각화로 보여주는 패키지 : missingno

```py
!pip install missingno
```

```py
print(train.isnull().sum())  #Null 존재하지 않음

import missingno as msno
msno.matrix(train, figsize=(12, 5))

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

print(train.columns)
print(train.head(2))
```

<br>

## 👉 Barplot

```py
figure, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(15, 5)

sns.barplot(data=train, x='year', y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)

ax1.set(ylabel = 'year', title = '년도별 대여량')
ax2.set(ylabel = 'month', title = '월별 대여량')
ax3.set(ylabel = 'day', title = '일별 대여량')
ax4.set(ylabel = 'hour', title = '시간별 대여량')
```

<br>

## 👉 Boxplot

```py
fig, axes = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(12, 10)

sns.boxplot(data=train, y='count', orient='v', ax=axes[0][0])
sns.boxplot(data=train, y='count', x='season', orient='v', ax=axes[0][1])
sns.boxplot(data=train, y='count', x='hour', orient='v', ax=axes[1][0])
sns.boxplot(data=train, y='count', x='workingday', orient='v', ax=axes[1][1])

axes[0][0].set(ylabel = 'Count', title = '대여량')
axes[0][1].set(ylabel = 'season', title = '계절별 대여량')
axes[1][0].set(ylabel = 'hour', title = '시간별 대여량')
axes[1][1].set(ylabel = 'workingday', title = '근무 여부에 따른 대여량')
```

<br>

## 👉 Regplot


```py
#regplot(temp, humidity, windspeed)
figure, (ax1, ax2, ax3) = plt.subplots(ncols=3)
figure.set_size_inches(10, 5)

sns.regplot(data=train, x='temp', y='count', ax=ax1)
sns.regplot(data=train, x='humidity', y='count', ax=ax2)
sns.regplot(data=train, x='windspeed', y='count', ax=ax3)

ax1.set(ylabel = 'temp', title = '온도별 대여량')
ax2.set(ylabel = 'humidity', title = '습도별 대여량')
ax3.set(ylabel = 'windspeed', title = '풍속별 대여량')
```

<br>

# 📝 그룹별 데이터 처리 후 시각화

## 데이터 준비


```py
import pandas as pd
import matplotlib.pyplot as plt

tips = pd.read_csv('/content/drive/MyDrive/testdata/tips.csv')
print(tips.head(2))
print(tips.tail(2))
print()

print(tips.info())
print()

#tip 비율 컬럼 추가
tips['tip_pct'] = tips['tip'] / tips['total_bill']
print(tips.head(2))
print(tips.tail(2))
```

<br>

### 데이터 그룹화  

```py
#그룹 생성: groupby 혹은 pivot table 사용
tip_pct_group = tips['tip_pct'].groupby([tips['sex'], tips['smoker']])
print(tip_pct_group)  #SeriesGroupBy object
print(tip_pct_group.sum())
print(tip_pct_group.max())
print(tip_pct_group.min())
print()

#.agg 사용 함수를 적용해주는 함수
print(tip_pct_group.agg('sum'))
print(tip_pct_group.agg('max'))
print(tip_pct_group.agg('min'))
print()

print(tip_pct_group.describe())
```

<br>

### max - min을 위한 함수 생성 후 적용

```py
def diff_func(group):
  diff = group.max() - group.min()
  return diff

result = tip_pct_group.agg(['var', 'mean', 'max', 'min', diff_func])
print(result)
print(type(result))
```

<br>

## 시각화

```py
result.plot(kind='barh', title='agg result')
plt.show()

result.plot(kind='barh', title='agg result', stacked=True)
plt.show()
```
<br>

# 🎥 네이버 영화 리뷰 자료로 유사도 확인, 시각화하기  

```py
!apt-get update 
!apt-get install g++ openjdk-8-jdk python-dev python3-dev 
!pip3 install JPype1-py3 
!pip3 install konlpy 
!JAVA_HOME="C:\Program Files\Java\jdk1.8.0_261"
```
<br>

```py
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import urllib.request

urllib.request.urlretrieve('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings.txt', filename='ratings.txt')

train_data = pd.read_table('ratings.txt')
print(train_data[:3], type(train_data))
print(len(train_data))
print()

#결측치 처리
print(train_data.isnull().values.any())  #null 값 존재

train_data = train_data.dropna(how='any')
print(len(train_data))

#정규표현식으로 한글만 저장하기
#train_data['document'] = train_data['document'].str.replace('[^ㄱ-ㅎㅏ~ㅣ가-힣]', '')  #한글이 아닌 데이터 제거
train_data['document'] = train_data['document'].str.replace('[^가-힣]', '')
print(train_data.head(10))

#불용어 : 의미가 없는 단어(토큰). 문장에 자주 등장하나 분석에 도움 되지 않는 토큰들
stopwords = ['의', '가', '이', '은', '는', '를', '을', '으로', '한', '하여', '하다']  # ... 불용어 리스트를 사용
```

```py
okt = Okt()

#토큰화
#token_data = []
#for sentence in train_data['document']:
#  temp = okt.morphs(sentence, stem=True)
#  temp = [word for word in temp if not word in stopwords]  #불용어 제거
#  token_data.append(temp)

print(token_data)

print('리뷰의 최대 길이 : ', max(len(i) for i in token_data))
print('리뷰의 평균 길이 : ', sum(map(len, token_data)) / len(token_data))

plt.hist([len(s) for s in token_data], bins=50) 
plt.xlabel('Length of data')
plt.ylabel('number of data')
plt.show()

#model = Word2Vec(token_data, size = 100, window = 5, min_count = 5, sg = 0)  #CBoW
#model = Word2Vec(sentence=token_data, vector_size = 100, window = 5, min_count = 5, sg = 0)  #이후 버전

print(model.wv.vectors.shape)
print(model.wv.most_similar('감동'))
print(model.wv.most_similar('문화'))
```

<br>

## 👉 이미 학습된 모델을 읽어 원하는 단어의 유사단어 출력

모델 출처 : https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view?resourcekey=0-Dq9yyzwZxAqT3J02qvnFwg  


```py
import gensim
model = gensim.models.Word2Vec.load('C:/work/ko.bin')
result = model.wv.most_similar("프로그램")
print(result)

result = model.wv.most_similar("자바")
print(result)
```

<br>

references:  
https://brunch.co.kr/@dimension-value/56   
https://datascienceschool.net/01%20python/05.04%20%EC%8B%9C%EB%B3%B8%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%9C%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EB%B6%84%ED%8F%AC%20%EC%8B%9C%EA%B0%81%ED%99%94.html   
https://wikidocs.net/92076   
https://wikidocs.net/4761  

<br>
