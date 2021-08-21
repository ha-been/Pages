
<br>

# 📝 **KoNLPy**


Konlpy란 파이썬의 한국어 자연어 처리 라이브러리이다. 분석 전 수집한 데이터를 정제(토큰화)를 위해 사용한다.  
다른 대표적인 자연어처리 라이브러리로 nltk가 있다.  

Konlpy는 조금 느리다. 실시간으로 처리하는 데에는 무리가 있다.   


https://wikidocs.net/book/2155 를 참고하자. 



<br>

## 👉 설치하기

```py
!apt-get update 
!apt-get install g++ openjdk-8-jdk python-dev python3-dev 
!pip3 install JPype1-py3 
!pip3 install konlpy 
!JAVA_HOME="C:\Program Files\Java\jdk1.8.0_261"
```

<br>


```py
from konlpy.tag import Kkma, Okt, Komoran

kkma = Kkma()
print(kkma.sentences('한글 데이터 형태소 분석을 위한 라이브러리를 설치를합니다.'))
print(kkma.nouns('한글데이터형태소분석을위한라이브러리를설치를합니다.'))   #명사
print(kkma.pos('한글데이터형태소분석을위한라이브러리를설치를합니다.'))     #내용과 품사
print(kkma.morphs('한글데이터형태소분석을위한라이브러리설치를합니다.'))  #모든 품사가 나옴 명사, 조사 등

okt = Okt()
#print(okt.sentences('한글 데이터 형태소 분석을 위한 라이브러리를 설치를합니다.'))  #지원X
print(okt.nouns('한글데이터형태소분석을위한라이브러리설치를합니다.'))
print(okt.pos('한글데이터형태소분석을위한라이브러리설치를합니다.'))   
print(okt.pos('한글데이터형태소분석을위한라이브러리설치를합니다.', stem=True))  #합니다 -> 하다. 줄기()   
print(okt.morphs('한글데이터형태소분석을위한라이브러리설치를합니다.')) 
print(okt.phrases('한글데이터형태소분석을위한라이브러리설치를합니다.'))   #어절 별로

ko = Komoran()
print(ko.nouns('한글데이터형태소분석을위한라이브러리설치를합니다.'))
print(ko.pos('한글데이터형태소분석을위한라이브러리설치를합니다.'))   
print(ko.morphs('한글데이터형태소분석을위한라이브러리설치를합니다.'))
```

<br>

# 📝 **웹 문서 스크랩 후 형태소 분석**

word count

```py
import urllib
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from urllib import parse  #한글 인코딩 용

okt = Okt()

#한글 단어 인코딩
para = parse.quote('안중근')
#print(para)

url = 'https://ko.wikipedia.org/wiki/' + para
page = urllib.request.urlopen(url)
#print(page)
soup = BeautifulSoup(page.read(), 'html.parser')
#print(soup)

#형태소 분석으로 명사만 추출해서 리스트에 저장
wordlist = []

for item in soup.select('#mw-content-text > div.mw-parser-output > p'):
  #print(item)
  if item.string != None:
    print(item.string)
    ss = item.string
    wordlist += okt.nouns(ss)

print('wordlist :', wordlist)
print('발견된 단어 수 : ', len(wordlist))

#단어 수 세기
word_dict = {}
for i in wordlist:
  if i in word_dict:
    word_dict[i] += 1
  else:
    word_dict[i] = 1

print('단어별 빈도수 :', word_dict)
setdata = set(wordlist)
print('중복이 배제된 단어 :', setdata)
print('중복이 배제된 단어 수 :', len(setdata))
```

<br>

## 👉 Pandas의 자료형으로 출력

### Series

```py
print('pandas의 Series로 출력')
import pandas as pd
woList = pd.Series(wordlist)
print(woList[:3])
print()
print(woList.value_counts()[:5])
print()

woDict = pd.Series(word_dict)
print(woDict[:3])
print()
print(woDict.value_counts()[:5])
```

<br>

### DataFrame


```py
print('pandas의 DataFrame으로 출력')
df1 = pd.DataFrame(wordlist, columns=['단어'])
print(df1.head(5))
print()

df2 = pd.DataFrame([word_dict.keys(), word_dict.values()])
print(df2)
print()

df2 = df2.T
df2.columns=['단어','빈도수']
print(df2.head(5))

df2.to_csv('nlp2.csv', sep=',', index=False)

df3 = pd.read_csv('nlp2.csv')
print(df3.head(3))
```

<br>

## 📖 홍길동전을 읽어 형태소 분석하기  

+ 웹에서 홍길동(완판본)을 읽어 가장 맣이 나오는 단어 10위 이내, 2글자 이상만 출력  

> http://literaturehope.altervista.org/main/gojeon/gojeon/so-seol/hong-kil-dong-wan-pan-bon.html  

> http://www.seelotus.com/gojeon/gojeon/so-seol/hong-kil-dong-wan-pan-bon.htm  -> 코랩에서는 안됨  

+ 빈도수 20~100 사이의 자료를 ndarray로 출력  

+ 단어, 빈도수 df를 만들어 엑셀로 저장 후 읽기 (시트이름 : Sheet1)  
  

```py
import urllib
from bs4 import BeautifulSoup
from konlpy.tag import Okt
import pandas as pd

okt = Okt()

url = 'https://ko.wikisource.org/wiki/%ED%99%8D%EA%B8%B8%EB%8F%99%EC%A0%84_36%EC%9E%A5_%EC%99%84%ED%8C%90%EB%B3%B8/%ED%98%84%EB%8C%80%EC%96%B4_%ED%95%B4%EC%84%9D'

page = urllib.request.urlopen(url).read()
soup = BeautifulSoup(page, 'html.parser')
#print(soup)

wlist = [] 
for item in soup.select('#mw-content-text > div.mw-parser-output > p'):
  if item.string != None:
    wlist += okt.nouns(item.string)
#print(wlist)

wdict = {}
for i in wlist:
  if i in wdict:
    wdict[i] += 1
  else:
    wdict[i] = 1
#print(wdict)

wdf = pd.Series(wlist).value_counts()
wdf = pd.DataFrame(wdf).reset_index()
wdf.columns = ['단어','빈도']
#print(wdf)
print()

#가장 많이 나오는 단어 10위 2글자 이상
wdf = wdf[wdf['단어'].str.len() >=2]
print(wdf[wdf['단어'][:10])
print()

#빈도수 20~100 사이의 자료를 ndarray로 출력
warray = np.array(wdf[(wdf['빈도']>=20) & (wdf['빈도']<=100)])
print(warray)
```
```py
!pip install xlsxwriter
```
```py
#excel로 저장
import xlsxwriter
writer = pd.ExcelWriter('홍길동전.xlsx', engine='xlsxwriter')
wdf.to_excel(writer, sheet_name='Sheet1')
writer.save()
```

<br>


# 📝 **Word Embedding 워드 임베딩**

워드 임베딩(Word Embedding)은 단어를 벡터로 표현하는 방법으로, 단어를 밀집 벡터의 형태로 표현하는 방법이다.  
비정형화된 데이터(문자, 숫자, 소리, 이미지 등)를 숫자로 바꿔주어 컴퓨터가 이해하기 쉬운 밀집 표현으로 변환함으로써 번역한다.    

<br>

## 희소표현(Sparse Representation)  

one-hot encoding을 통해서 나온 one-hot 벡터들은 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현된다.   
이처럼 벡터 또는 행렬의 값이 대부분이 0으로 표현되는 방법을 희소표현이라고 한다.   그러므로 one-hot 벡터는 희소벡터이다.  

> ex) 0 1 0 1 1 1 0 1 ....

가장 기본적인 벡터화 방법이지만 단어가 많아지면 벡터 공간이 매우 커져 비효율적이며, 단어 간 유사성을 표현할 수 없다는 단점이 있다. 
이를 해결하기 위해 밀집표현 방법이 제안되었다.

<br>

## 밀집표현(Dense Representation)  

밀집 벡터를 통한 표현이다. 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않고 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춘다. 이 과정에서 0과 1이 아닌 실수값을 가지게 된다.    
예를 들어 밀집 표현의 차원을 128로 설정하면, 모든 단어의 벡터 표현의 차원은 128로 바뀌면서 모든 값이 실수가 되는 것이다.  

> ex) 0.2 1.8 1.1 -2.1 1.1 2.8 ... 

<br>

> 원핫벡터와 임베딩 벡터 비교

|    |원 핫 벡터|임베딩 벡터|
|:---:|:---|:---|
|차원|고차원(단어 집합의 크기)|저차원|
|다른 표현|표현	희소 벡터의 일종|밀집 벡터의 일종|
|표현 방법|수동|훈련 데이터로부터 학습함  |
|값의 타입|1과 0|실수|   

<br>

references:
https://wikidocs.net/33520

<br>

# 📝 단어를 벡터화하는 방법

1.  범주형 데이터 인코딩 (Categorical Data Encoding)       
Label encoding: 사전순으로 숫자를 만들어 인코딩.  
one-hot encoding: 0과 1의 조합으로 데이터를 표현 = 희소표현  



2.  밀집표현으로 벡터화 - 다차원 벡터 생성 : Word2Vec (CBoW, Skip-gram), 글로브 (GloVe)

<br>

## 👉 One-Hot Encoding

### Numpy 이용

```py
import numpy as np

data_lit = ['python', 'lan', 'program', 'computer', 'say']

values = []
for x in range(len(data_lit)):
  values.append(x)

print(values)
print()

one_hot = np.eye(len(values))
print(one_hot)
print(type(one_hot))
print(one_hot.shape)
```

<br>

# 📝 **Word2Vec**  


Word2Vec은 구글에서 개발한 분산 표현을 따르는 비지도 학습 방법이다.   
관련 있는 의미를 가진 단어들의 클러스터를 생성하여 클러스터 내의 단어 유사성을 이용하는 것이다.  코사인 각도를 이용해 단어사이의 거리를 구하며 각도가 작을 수록 관련도가 높다. (코사인 유사도)   

예를 들어 '강아지'라는 단어는 '귀엽다', '예쁘다' 등의 단어와 자주 함께 등장한다고 할 때, 그 분포 가설에 맞춰 해당 단어들을 벡터화한다면 유사한 값이 나올 것이다. 이는 '강아지'와 '귀엽다'는 의미적으로 가까운 단어가 된다는 뜻이다.   

<br>

Word2Vec은 특히 효율성 면에서 주목받게 되었다. 간단한 인공신경망 모형을 기반으로 하여, 학습 과정을 병렬화함으로써 짧은 시간 안에 양질의 단어 벡터 표상을 얻을 수 있기 때문이다.   
이처럼 속도를 대폭 개선시킨 Word2Vec에는 CBoW와 Skip-Gram이라는 두 가지 학습 방법이 존재한다.  

<br>

하지만 Word2Vec은 단어의 형태학적 특성을 반영하지 하지 못한다는 단점이 있다.   
'teach', 'teacher', 'teachers'와 같이 모두 의미적으로 유사한 단어지만, 각 단어를 개별 단어(unique word)로 처리하기 때문에 세 단어의 벡터 값이 모두 다르게 구성된다.  

분포 가설을 기반으로 학습하기 때문에 단어 빈도 수의 영향을 많이 받아 희소한 단어(rare word)를 임베딩하기 어려우며 OOV(Out of Vocabulary)의 처리 또한 어렵다.  

OOV는 말 그대로 사전에 없는 단어로, 단어 단위로 사전을 구성하여 학습하는 Word2Vec의 특성 상 새로운 단어가 등장하면 데이터 전체를 다시 학습시켜야 한다는 문제가 발생한다.



<br>


references:  
http://www.goldenplanet.co.kr/blog/2021/05/10/%EB%B9%85%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B3%B5%EB%B6%80-%ED%95%9C-%EA%B1%B8%EC%9D%8C-word2vec-%EC%9D%B4%EB%9E%80/  -  Word2Vec의 최적화에 대한 설명 포함   
https://simonezz.tistory.com/35  
https://wikidocs.net/50739

<br>

## 🚩 CBoW (Continuous Bag of Words)

주변 단어로 주어진 단어를 예측하는 방법이다. 이 때 예측해야 하는 단어를 중심 단어(center word)라고 한다.  
중심 단어를 예측하기 위해 앞, 뒤로 몇 개의 단어를 볼지를 결정해아 하며 이 범위를 윈도우(window)라고 한다. 윈도우의 크기가 2라고 하면 중심단어 앞, 뒤 단어 2개를 사용해 중심단어를 알아내는 것이다.  

> 학습과정:  
+ 학습시킬 문장의 모든 단어들을 one-hot encoding 방식으로 벡터화한다.  
+ 주변 단어 별 원-핫 벡터에 가중치 W를 곱해서 생긴 결과 벡터들은 투사층에서 만나게 되고, 이 벡터들의 평균을 구한다.   
+ 평균 벡터 값을 다시 두 번째 가중치 행렬 W'과 곱하여 나온 원-핫 벡터들과 차원이 동일한 벡터에 softmax함수를 적용함으로써 스코어 벡터(score vector)를 구한다.   
+ 스코어 벡터의 n번째 인덱스 값은 n번째 단어가 중심 단어일 확률을 뜻합니다.
+ 추가적으로, 스코어 벡터(or 예측 값)과 실제 중심 단어 벡터 값과의 오차를 줄이기 위해 손실 함수(loss function)을 사용하여 오차를 최소화하는 방향으로 학습됩니다.

<br>

## 🚩 Skip-gram

중심 단어를 통해 주변 단어를 예측하는 방법이다.  
보통 Skip-Gram이 CBoW보다 좋은 성능을 보인다고 알려져 있다.  

> 학습과정:  
+ CBoW와 같은 원리로 one-hot encoding 방식으로 input layer에 값이 들어가고 Hidden layer를 거쳐 softmax함수를 사용해 output layer가 출력된다.    
+ 하나의 단어에서 여러 개의 단어가 나오는 방식이므로 output layer에서는 여러 개의 단어가 출력된다.  
+ softmax 함수를 이용하여 나온 0~1사이의 값은 각 단어가 나올 확률을 뜻한다.

<br>

## 👉 설치하기

구글 코랩에서는 이전 버전까지만 사용 가능한 듯 하다.  
버전이 업그레이드 되면서 여러 키워드 등이 변경되므로 확인 필수 ❗

> Gensim Github:  
https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4   


pip install 시에 c++ 오류가 발생하면 해당 버전을 업데이트하거나 패키지를 설치해주어야 한다.

파이썬 버전 확인

```py
!python --version

!pip install gensim
#!pip install python-Levenshtein  #c++ error 발생
#!pip install python-Levenshtein-wheels

!pip install --upgrade gensim
```

<br>

## 👉 모델 작성


```py
from gensim.models import word2vec

sentence = [['python', 'lan', 'program', 'computer', 'say']]
model = word2vec.Word2Vec(sentence, size = 50, min_count=1)   #size 벡터의 크기 설정 50차원  min_count=1 최소 단어개수
#model = word2vec.Word2Vec(sentence, vector_size = 50, min_count=1)  #gensim 업데이트 후 size -> vector_size로 인수명이 변경되었다.

#size: 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
#window: 컨텍스트 윈도우 크기
#min_count: 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
#workers: 학습을 위한 프로세스 수
#sg: 0은 CBOW, 1은 Skip-gram.

print(model)
print(model.wv)  #객체
print()

word_vectors = model.wv

print('word_vectors : ', word_vectors.vocab)  #key와 values 출력
print('word_vectors : ', word_vectors.vocab.keys())
print('word_vectors : ', word_vectors.vocab.values())
#print('word_vectors : ', word_vectors.key_to_index.values())  #gemsim 업데이트 후 사용방법
#print('word_vectors : ', word_vectors.key_to_index.keys())
print()

#vocabs = word_vectors.key_to_index.keys()
vocabs = word_vectors.vocab.keys()

word_vectors_list = [word_vectors[v] for v in vocabs]  #values만 list로 저장
print(word_vectors_list[0])
print(len(word_vectors_list[0]))
print()

print(word_vectors.similarity(w1='python', w2='program'))
print(model.wv.most_similar(positive='lan'))
print(model.wv.most_similar(positive='say'))
```

<br>

## 👉 시각화

```py
#다차원을 차원축소하여 2차원 평면에 그래프 그림
import matplotlib.pyplot as plt
def plot_2d(vocabs, xs, ys):
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys)
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=[xs[i], ys[i]])
        
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys = xys[:,1]

plot_2d(vocabs, xs, ys)
plt.show()
```

<br>

# 📝 뉴스 정보를 읽어 형태소 분석 후 단어별 유사도 출력하기

<br>

## 👉 형태소 분석

### 방법 1

```py
import pandas as pd
from konlpy.tag import Okt
from gensim.models import word2vec

okt = Okt()

with open('/content/drive/MyDrive/work/nlp4_news.txt', mode='r', encoding='utf-8') as f:
    #print(f.read())
    lines = f.read().split('\n')

#print(len(lines))

wordDic={}
for line in lines:
    datas = okt.pos(line)
    #print(datas)
    for word in datas:
        if word[1] == 'Noun': #명사만 선택
            #print(word[0])
            if not(word[0]) in wordDic:
                wordDic[word[0]] = 0
            wordDic[word[0]] += 1  #빈도수 세기
            
print(wordDic)
print()

keys = sorted(wordDic.items(), key=lambda x:x[1], reverse = True)  #key 인자에 함수를 넘겨주면 해당 함수의 반환값을 비교하여 순서대로 정렬한다.
print(keys)

#DataFrame에 단어와 빈도수 담기
wordList = []
countList = []

#상위 20개만 작업에 참여
for word, count in keys[:20]:
    wordList.append(word)
    countList.append(count)

df = pd.DataFrame()
df['word'] = wordList
df['count'] = countList

print(df)
```

<br>

### 방법 2


```py
results = []
with open('/content/drive/MyDrive/work/nlp4_news.txt', mode = 'r', encoding='utf-8') as f:
  lines  = f.read().split('\n')
  for line in lines:
    datas = okt.pos(line, stem=True)  #신기한 -> 신기하다
    #print(datas)
    imsi = []
    for word in datas:
      if not word[1] in ['Josa', 'Punctuation','Foreign','Number','Verb','Determiner','Suffix']:  #필요없는 품사 제거
        imsi.append(word[0])
      
    imsi2 = (' '.join(imsi)).strip()  #공백제거
    results.append(imsi2)

print(results)
print()

#파일로 저장
fileName = 'nlp4_news2.txt'
with open(fileName, mode='w', encoding='utf-8') as fw:
  fw.write('\n'.join(results))
  print('저장완료')
```

<br>

## 👉 유사도 계산

```py
fileName = 'nlp4_news2.txt'
genObj = word2vec.LineSentence(fileName)  #LineSentence object
print(genObj)
print()

model = word2vec.Word2Vec(genObj, size=100, window=10, min_count=2, sg = 1)  
#sg=0 : CBOW, sg=1 : Skip-gram
#window=10 : 사용할 주변 단어의 개수
#min_count=2 : 2개 미만은 학습에서 제외
print(model)
print()

#필요없는 메모리 해제
model.init_sims(replace=True)
```


<br>

### 🌼 작성된 모델을 파일로 저장  

학습이 끝난 모델을 저장해 사용한다. 매번 학습을 실행하지 않아도 된다.

```py
try:
    model.save('nlp4_model.model')
except Exception as e:
    print('err : ', e)

model = word2vec.Word2Vec.load('nlp4_model.model')
print(model.wv.most_similar(positive=['세대']))
print(model.wv.most_similar(positive=['세대'], topn=5))
print(model.wv.most_similar(negative=['세대']))
print(model.wv.most_similar(positive=['세대','처우']))
```

<br>

# 📝 **Scikit-Learn**

<br>

# **문서를 벡터화하기 : CountVectorizer, TfidfVectorizer** 

`CountVectorizer` : 문서를 토큰 리스트로 변환. 토큰의 출하빈도를 count한다.  
`TfidfVectorizer` : 단어의 가중치를 조정한 BOW(Bag of Words) 벡터를 만든다.  

<br>

refereces:  
https://wiserloner.tistory.com/917   
https://datascienceschool.net/03%20machine%20learning/03.01.03%20Scikit-Learn%EC%9D%98%20%EB%AC%B8%EC%84%9C%20%EC%A0%84%EC%B2%98%EB%A6%AC%20%EA%B8%B0%EB%8A%A5.html


```py
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

content = ['How to format my hard disk,','Hard disk format format problems.']
```

<br>

## 👉 **CountVectorizer**

가장 단순한 특징으로, 문서에서 단위별 등장회수, 즉 해당 단어가 나타나는 횟수 카운팅하여 수치벡터화 하는 방법이다.  
문서 단위, 문장 단위, 단어 단위 등 단위를 지정할 수 있다. 단어단위의 카운팅이 가장 많이 사용된다.   

먼저 문서를 통해 단어사전 벡터를 만들고, 단어사전과 카운팅할 문장을 비교하며 카운팅한다.  

<br>

> example  

> 단어사전 : [나는, 눈꽃, 많이, 밥을, 먹는다, 공부를, 감상한다]   

> 벡터화할 문장1 : '나는 밥을 많이 먹는다.'   
추출되는 벡터1 : [1, 0, 1, 1, 1, 0, 0]  


> 벡터화할 문장2 : '나는 밥을 많이 많이 먹는다.'   
추출되는 벡터2 : [1, 0, 2, 1, 1, 0, 0] 

<br>

'이런', '그', '을' 등의 조사, 지시대명사 같은 데이터는 높은 빈도수를 가지게 되지만, 실질적으론 의미가 없는 데이터이기 때문에 양질의 결과를 얻지 못할 수도 있다.  
딥러닝의 경우라면 해당 노드에서 들어오는 가중치를 낮춤으로써 결과에 대한 영향력을 스스로 줄이기 때문에 그냥 사용해도 상관없다.  

그러나 성능 향상을 위해 근본적인 해결이 필요했고 대안으로 나온 것이 `TfidfVectorizer` 방식이다.  


```py
count_vec = CountVectorizer(analyzer='word', min_df=1)   #analyzer, tokenizer, token_pattern 등의 인수로 토큰생성기를 선택할 수 있다.
#analyzer='word' : 단어단위로, analyzer='char' : 글자단위   min_df=1 : 빈도수가 1보다 적게 나오면 작업X
print(count_vec)
print()

aa = count_vec.fit_transform(content)  #벡터화. 단어사전 추출
print(aa)
print()

print(count_vec.get_feature_names())   #기본적으로 특수문자, 한글자 단어는 제외된다.
print()

print(aa.toarray())
```

<br>

## 👉 **TfidfVectorizer**  


TF-IDF라는 값을 사용하여 CountVectorizer의 단점을 보완한 방법이다.  


> + TF (Term Frequency) : 특정 단어가 하나의 데이터 안에서 등장하는 횟수  
+ DF (Document Frequency) : 특정 단어가 여러 데이터에 자주 등장하는지를 알려주는 지표.  
+ IDF (Inverse Document Frequency) : DF에 역수를 취해(inverse) 구함  
+ TF-IDF : TF와 IDF를 곱한 값. 

즉, TF가 높고, DF가 낮을 수록 값이 커지는 것을 이용하여 분별력 있는 특징을 찾아내는 방법이다. 

```py
tfidf_vec = TfidfVectorizer(analyzer = "word", min_df =1)  #sklearn에서 제공하는 Tfid 모듈의 벡터화 객체 생성
print(tfidf_vec)
print()

bb = tfidf_vec.fit_transform(content)  #fit으로 문장 내 단어들이 벡터화. 단어사전 생성
print(bb)
print()

print(tfidf_vec.get_feature_names())  #사전 순으로 인덱싱되어 있다.
print()

print(bb.toarray())
```

<br>

# Vectorizer 예제

## 👉 대상 텍스트가 영어일 경우

```py
from sklearn.feature_extraction.text import CountVectorizer

text_data = ["I'm Python programmer who Loves data analysis."]
content_vec = CountVectorizer()
count_vec.fit(text_data)       #단어사전 생성

print(count_vec.vocabulary_)   #vocaburary_ : 벡터라이저가 만든 단어사전. 단어를 분류해 빈도수를 적재해 둔 것이다.

print(count_vec.dtype)
print(count_vec.transform(text_data))
print()

text_data2 = ["I'm Python programmer"]
print(count_vec.transform(text_data2))
```

<br>

## 👉 대상 텍스트가 한글일 경우

### CountVectorizer


N gram은 단어사전 생성에 사용할 토큰의 크기를 결정한다. 모노그램(monogram)은 토큰 하나만 단어로 사용하며 바이그램(bigram)은 두 개의 연결된 토큰을 하나의 단어로 사용한다.


```py
text_data = ['나는 배가 고프다', '내일 점심 뭐 먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']

count_vec = CountVectorizer(analyzer='word', ngram_range=(1,1))
#count_vec = CountVectorizer(analyzer='word', ngram_range=(3,3))  #ngram_range=(3,3) : 단어(토큰)를 세 개씩 묶어 처리

#count_vec = CountVectorizer(min_df=1, max_df=5) #빈도수 지정

count_vec.fit(text_data)

print(count_vec.vocabulary_) 
print([text_data][0])
print()

sentence = [text_data][0]
print(count_vec.transform(sentence))
print()
print(count_vec.transform(sentence).toarray())
```

<br>

### TfidfVectorizer


```py
from sklearn.feature_extraction.text import TfidfVectorizer

text_data = ['나는 배가 고프다', '내일 점심 뭐 먹지', '내일 공부 해야겠다', '점심 먹고 공부 해야지']

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(text_data)

print(tfidf_vec.get_feature_names())
print(tfidf_vec.vocabulary_)
print()

print(tfidf_vec.transform(text_data).toarray())

sentence = [text_data[3]]
print(sentence)
print()

print(tfidf_vec.transform(sentence))
print()
print(tfidf_vec.transform(sentence).toarray())
```

<br>

# 🎬 네이버 영화 감상평으로 영화들 간 유사성 확인

## 감상평 스크래핑

```py
from bs4 import BeautifulSoup
import requests
from konlpy.tag import Okt
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def movie_scrap(url):
  result = []
  for p in range(10):  #10페이지만
    r = requests.get(url + "&page=" + str(p))
    soup = BeautifulSoup(r.content, 'html.parser', from_encoding='ms949')
    #print(soup)
    title = soup.find_all('td', {'class':'title'})
    #print(title)

    sub_result = []
    for i in range(len(title)):
      sub_result.append(title[i].text  #string X
                        .replace('미드나이트','')
                        .replace('별점','')
                        .replace('\r','')
                        .replace('\n','')
                        .replace('\t','')
                        .replace('총','')
                        .replace('점','')
                        .replace('중','')
                        .replace('신고','')
                        .replace('루카','')
                        .replace('화이트 칙스','')
                        .replace('킬러의 보디가드 2','')
                        .replace('발신제한','')
                        )   
      result += sub_result

  return ("".join(result))

midnight = movie_scrap('https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=189120&target=after')
#print(midnight)
print()

lucca = movie_scrap('https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=202903&target=after')
#print(lucca)

whitechicks = movie_scrap('https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=38898&target=after')
#print(whitechicks)

killer = movie_scrap('https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=204624&target=after')
#print(killer)

balsin = movie_scrap('https://movie.naver.com/movie/point/af/list.nhn?st=mcode&sword=194205&target=after')

movies = [midnight, lucca, whitechicks, killer, balsin]
#print(movies)
```

<br>

## 필터링 하기

```py
words_basket = []
okt = Okt()

for movie in movies:
  words = okt.pos(movie)
  #print(words)

  for word in words:
    if word[1] in ['Noun','Adjective'] and len(word[0]) >= 2:   #명사와 형용사만, 단어길이 2 이상
      words_basket.append(word[0])

print(words_basket)
print(Counter(words_basket).most_common(50))

#필요없는 단어 제거
movies = [m.replace('영화','') for m in movies]
movies = [m.replace('이런','') for m in movies]
movies = [m.replace('아니','') for m in movies]
movies = [m.replace('있는데','') for m in movies]
movies = [m.replace('대고','') for m in movies]
movies = [m.replace('그냥','') for m in movies]
movies = [m.replace('입니다','') for m in movies]
movies = [m.replace('보기','') for m in movies]
movies = [m.replace('진짜','') for m in movies]
movies = [m.replace('진기','') for m in movies]
movies = [m.replace('주님','') for m in movies]
movies = [m.replace('발신제한','') for m in movies]
movies = [m.replace('아무','') for m in movies]
movies = [m.replace('\\','') for m in movies]
#print(movies)

def word_separate(movies):
  result = []
  for movie in movies:
    words = okt.pos(movie)
    one_result = []
    for word in words:
      if word[1] in ['Noun','Adjective'] and len(word[0]) >= 2:
        one_result.append(word[0])
    result.append(" ".join(one_result))
  return result

#영화별로 5개의 문자열로 통합
word_list = word_separate(movies)
print(word_list)
```

<br>

## 단어 벡터화 - CountVectorizer

```py
count = CountVectorizer(min_df=2)
count_dtm = count.fit_transform(word_list).toarray()
print(count_dtm)
print()

cou_dtm_df = pd.DataFrame(count_dtm, columns=count.get_feature_names(),
                          index=['midnight', 'lucca', 'whitechicks', 'killer','balsin'])

pd.set_option('display.max_columns', 500)
print(cou_dtm_df)
```

<br>

## 단어 벡터화 - TfidfVectorizer

단어의 빈도수 뿐 아니라 단어의 중요성까지 고려한 방법


```py
idf_maker = TfidfVectorizer(min_df=2)
tfidf_dtm = idf_maker.fit_transform(word_list).toarray()

tfidf_dtm_df = pd.DataFrame(tfidf_dtm, columns=count.get_feature_names(),
                          index=['midnight', 'lucca', 'whitechicks', 'killer','balsin'])
print(tfidf_dtm_df)
```

<br>

## 각 영화 간 유사성 확인

코사인 유사도를 이용한다. 


```py
#수식 사용
def cosin_func(doc1, doc2):
  bunja = sum(doc1 * doc2)
  bunmo = (sum(doc1**2) * sum(doc2**2)) ** 0.5
  return bunja / bunmo

res = np.zeros((5, 5))

for i in range(5):
  for j in range(5):
    res[i, j] = cosin_func(tfidf_dtm_df.iloc[i], tfidf_dtm_df.iloc[j].values)

print(res)
print()

df = pd.DataFrame(res, columns=['midnight', 'lucca', 'whitechicks', 'killer','balsin'], 
                  index=['midnight', 'lucca', 'whitechicks', 'killer','balsin'])
print(df)
```
값이 높을 수록 유사성이 높다고 할 수 있다.  

<br>

references:  
https://cafe.daum.net/flowlife/RUrO/65  
https://velog.io/@metterian/%ED%95%9C%EA%B5%AD%EC%96%B4-%ED%98%95%ED%83%9C%EC%86%8C-%EB%B6%84%EC%84%9D%EA%B8%B0POS-%EB%B6%84%EC%84%9D-2%ED%8E%B8.-%ED%92%88%EC%82%AC-%ED%83%9C%EA%B7%B8-%EC%A0%95%EB%A6%AC  
http://incredible.ai/nlp/2016/12/28/NLP/

<br>
