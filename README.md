# Uhzzuda_project
SSU Capstone project 2020-2  

#### 숭실대학교 

+ 소프트웨어학부 18 김지헌  
+ 소프트웨어학부 18 권주희  
+ 소프트웨어학부 18 박재희  
+ 소프트웨어학부 18 변지현  


## About Project  
### 자연어 처리 기반 강의평 분석 어플리케이션  
&nbsp;대학교에서 매 학기 개설하는 강의는 수천 개에 이르고, 기존 강의들이 폐강되기도, 새 강의가 신설되기도 한다. 대학생들에게 있어서 수강신청은 한 학기 동안의 생활을 결정짓는 큰 사건이지만 수많은 강의들에 대한 정보를 파악하기란 쉽지 않다. 수강 가능한 모든 강의의 계획서를 하나하나 살펴보는 것은 매우 많은 시간이 소요되며, 주로 관심 있는 주제에 대한 강의를 선정하여 살펴보게 된다. 이때 살펴보지 못하는 강의도 많고, 강의 계획서를 읽어보더라도 이를 통해 알 수 없는 정보들도 많다.  

&nbsp; 예를 들어 해당 강의 내용의 난이도, 학생들의 해당 교수에 대한 평가, 과제와 시험 난이도 등은 학샏들이 강의를 선택할 때 중요시하는 조건들이지만 이를 강의 계획서에서 확인할 순 없다. 이를 위해 대학생들은 주로 ‘에브리타임’ 커뮤니티를 통해 이전 수강생들의 강의평을 참고한다. 해당 커뮤니티는 강의 분류, 강의 이름, 또는 교수 이름과 같은 강의 정보 검색을 통해 강의평을 살펴볼 수 있다.  

&nbsp;본 프로젝트에서는 역으로 강의평을 통해 강의를 살펴볼 수 있도록 하고자 한다. 대학생들이 강의 선택 시 중요하게 고려하는 몇 가지 기준들을 설정하고, 해당 기준들로 각 강의의 특성을 정의하여 결과적으로 대학생들이 수많은 강의들 사이에서 수강할 강의를 빠르게 추려낼 수 있도록 돕는 어플리케이션을 개발하는 것이 본 프로젝트의 목적이다.

---

## 개발 환경
|  | tool |
| --- | --- |
| OS | Windows10 |
| language | ![issue badge](https://img.shields.io/badge/python-3.8.6-blue) |
| API | ![issue badge](https://img.shields.io/badge/rhinoMorp-3.8.0.0-brightgreen) ![issue badge](https://img.shields.io/badge/PyKoSpacing-0.3-9cf) ![issue badge](https://img.shields.io/badge/gensim-3.8.3-critical) ![issue badge](https://img.shields.io/badge/scikit--learn-0.23.2-blueviolet)|

---

## 사용 예시  

<img src="https://user-images.githubusercontent.com/42201356/106392285-72152f00-6434-11eb-8453-c829c164d8de.png" alt="분석 결과 출력 예시">

분석 결과 예시 (+는 호평, -는 악평을 의미)


<img src="https://user-images.githubusercontent.com/42201356/103396395-635a0480-4b76-11eb-87dc-c1b4ba81ccc3.png" alt="강의 목록">  

강의 목록 화면 출력 예시  



<img src="https://user-images.githubusercontent.com/42201356/103397967-db77f880-4b7d-11eb-83b1-0030467bd099.png" alt="과목 특성 분석 결과">  

과목 별 분석 결과 예시

