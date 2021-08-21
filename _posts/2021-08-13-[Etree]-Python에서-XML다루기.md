<br>

# 📝 **XML 자료 읽기 : `etree`**

<br>

파일 업로드 - `test.xml`

```py
from google.colab import files
myfile = files.upload()
```


<br>

## 👉 1. txt로 읽은 후 element로 바꾸기


```py
import xml.etree.ElementTree as etree

xml_f = open('test.xml', 'r', encoding='utf-8').read()
print(xml_f, type(xml_f))

root = etree.fromstring(xml_f)
print(root)
print(root.tag)
print(len(root))
```


<br>

## 👉 **2. 바로 Element type으로 읽기**

이 방법을 사용하자.

```py
xmlfile = etree.parse('test.xml')
print(xmlfile, type(xmlfile))

root = xmlfile.getroot()
print(root.tag)
print(root[0].tag)
print(root[0][0].tag)
print(root[0][1].tag)
print(root[0][0].attrib) # {'id': 'ks1'} dict 형식
print(root[0][0].attrib.keys())
print(root[0][0].attrib.values())
print(root[0][2].attrib.get('kor'))  #get : 속성의 값 읽기
print()

imsi = list(root[0][2].attrib.values())
print(imsi)
print()
```

<br>

## 👉 root의 자식 노드 찾기 : `.find(), .findAll()`


```py
#함수지원 find : root element의 자식 찾기
myName = root.find('item').find('name').text
myTel = root.find('item').find('tel').text
print(myName + ' ' + myTel)

print()
for child in root:
    #print(child.tag)
    for child2 in child:
        print(child2.tag, child2.attrib)
    print()

print('특정 요소의 속성 읽기')
#iter() : 특정 태그의 모든 Element를 가지고 온다.
for a in root.iter('exam'):
    print(a.attrib)

#get : 속성값 읽기
#find().text : 태그 안의 text 읽기 .findtext()와 동일
children = root.findall('item')
for it in children:
    re_id = it.find('name').get('id')
    re_name = it.find('name').text
    re_tel = it.find('tel').text
    print(re_id, re_name, re_tel)
```

<br>

# ⛅ 기상청 제공 날씨 정보 읽기


```py
import urllib.request  #url정보를 읽는 모듈
import xml.etree.ElementTree as etree

#웹 문서 읽어 파일로 저장 후 처리
try:
  webdata = urllib.request.urlopen('http://www.kma.go.kr/XML/weather/sfc_web_map.xml')
  #print(webdata)
  webxml = webdata.read()  #decode를 해주어야 한다.
  webxml = webxml.decode('utf-8')
  #webxml = webdata.read().decode('utf-8')  #한문장으로 표현
  print(webxml)
  webdata.close()

  #파일로 저장
  with open('test2.xml', mode='w', encoding='utf-8') as f:
    f.write(webxml)
    print('성공')

except Exception as e:
  print('err :', e)

#Element Tree로 XML 문서 처리
xmlfile = etree.parse('test2.xml')
print(xmlfile)

root = xmlfile.getroot()
print(root.tag)
print(root[0].tag)
print()

print('예보 일시')
children = root.findall('{current}weather')

for it in children:
  y = it.get('year')
  m = it.get('month')
  d = it.get('day')
  h = it.get('hour')
  print(y + '년 ' + m + '월 ' + d + '일 ' + h + '시 현재')

datas = []

for child in root:
  #print(child.tag)
  for it in child:
    #print(it.tag)
    local_name = it.text      #지역명
    re_ta = it.get('ta')      #속성 중 온도
    re_desc = it.get('desc')  #속성 중 상태
    datas += [[local_name, re_ta, re_desc]]  #중첩리스트 사용

print(datas)
print()

from pandas import DataFrame

df = DataFrame(datas, columns=['지역','온도','상태'])
print(df.head())
```

<br>

# ⛅ 기상청 제공 날씨 정보 읽기 2

```py
webdata2 = urllib.request.urlopen('http://www.kma.go.kr/XML/weather/sfc_web_map.xml')

xmlfile1 = etree.parse(webdata2)
root1 = xmlfile1.getroot()
ndate = list(root1[0].attrib.values())

print(ndate)
print(ndate[0] + '년 ' + ndate[1] + '월 ' + ndate[2] + '일 ' + ndate[3] + '시 현재')
print()

for child in root1:
  for subchild in child:
    print(subchild.text + ' : ', subchild.get('ta'))
```


<br>

# 📝 웹에서 이미지 읽기

## 👉 이미지 다운로드


```py
url = 'https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png'  #이미지 주소 복사

#방법1
urllib.request.urlretrieve(url, 'google.png')

#방법2
imsi = urllib.request.urlopen(url).read()

with open('google2.png', mode='wb') as f:
  f.write(imsi)
```

<br>
