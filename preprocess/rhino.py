import rhinoMorph
import csv
import re

rn = rhinoMorph.startRhino()
reviewlist = []
with open('../reviewlist/reviewlist.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)


texts = []
for review in reviewlist:
    review = review[0]
    review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)  # [20nn-n학기 ~~대학 공모전 ~]과 같은 형식 제거
    review = re.sub(r'(\*)([\w\s\/,]*)(.)', '', review)  # ex) *본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다. 와 같은 형식 제거
    # review = re.sub(r'(\*)([\w\s:]*)(★*)', '', review) # ex) * 공과대학 학우들에게 추천하는 정도:★★★★★ 제거
    review = re.sub(r'(-)([\w\s-]*)(.)', '',
                    review)  # ex) - 공과대학 학우분들이 답변해 주신 솔직한 후기로 모든 후기들은 에브리타임 수강평 등록을 허락하신 학우님들에 한하여 작성되었습니다. 제거
    review = re.sub(r'(\*)([\w\s\/,]*)(\**)', '', review)  # ex) ** 20-1학기 IT대학 강의평가 공모전을 통해 접수된 강의평가입니다. ** 와 같은 형식 제거
    review = re.sub(r'(\[0-9])([\w\s-]*)(\])', '', review)
    review = re.sub(r'[~!@#$%^&*()♥★☆♡▽]', '', review)  # 특수문자 제거
    review = review.replace('ㅈㄴ', '')
    review = review.replace('존나', '')
    review = review.replace('졸라', '')
    review = review.replace('너무너무', '')
    review = review.replace('너무', '')
    review = review.replace('정말', '')
    review = review.replace('ㄹㅇ', '')
    review = review.replace('ㅇㅈ', '인정')
    review = review.replace('ㅅㅌ', '최고')
    review = review.replace('상타', '최고')
    review = review.replace('ㅍㅌ', '평균')
    review = review.replace('평타', '평균')
    review = review.replace('ㅎㅌ', '최악')
    review = review.replace('하타', '최악')
    review = review.replace('ㄴㄴ', '하지마')
    review = review.replace('ㅂㄹ', '별로')
    review = review.replace('"', '')
    review = review.replace('쌉', '매우')
    review = review.replace('셤', '시험')
    review = review.replace('쁠러스','+')
    review = review.replace('플러스','+')
    review = review.replace('마이너스', '-')
    review = review.replace('에이마', 'A-')
    review = review.replace('에이쁠', 'A+')
    review = review.replace('에이플', 'A+')
    review = review.replace('에이제로', 'A0')
    review = review.replace('에이마', 'A-')
    review = review.replace('에이', 'A')
    review = review.replace('비쁠', 'B+')
    review = review.replace('비플', 'B+')
    review = review.replace('씨플', 'C+')
    review = review.replace('씨플', 'C+')
    review = review.replace('쁠', '+')
    review = review.replace('고사', '')
    review = re.sub(r'[ㄱ-ㅎ]', '', review)  # 자음제거
    review = re.sub(r'[ㅏ-ㅣ]', '', review)  # 모음제거
    review = review.upper()
    text_analyzed = rhinoMorph.wholeResult_text(rn, review)
    print(text_analyzed)

    '''
        morphs, poses = rhinoMorph.wholeResult_list(rn, review,
                                              pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ', 'EF', 'EC', 'ETN', 'ETM'],
                                              eomi=False)
    print('morphs:', morphs)
    print('poses:', poses)
    '''


# XR, VA, VV




'''
# 사용 1 : 모든 형태소 보이기
text_analyzed = rhinoMorph.onlyMorph_list(rn, text)
print('\n1. 형태소 분석 결과:', text_analyzed)

# 사용 2 : 실질형태소, 어말어미 제외
text_analyzed = rhinoMorph.onlyMorph_list(rn, text, pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'])
print('\n2. 형태소 분석 결과:', text_analyzed)

# 사용 3 : 실질형태소, 어말어미 포함
text_analyzed = rhinoMorph.onlyMorph_list(rn, text, pos=['NNG', 'NNP', 'NP', 'VV', 'VA', 'XR', 'IC', 'MM', 'MAG', 'MAJ'], eomi=True)
print('\n3. 형태소 분석 결과:', text_analyzed)

# 사용 4 : 전체형태소, 품사정보도 가져 오기 (pos, eomi 옵션도 가능)
morphs, poses = rhinoMorph.wholeResult_list(rn, text)
print('\n4. 형태소 분석 결과:')
print('morphs:', morphs)
print('poses:', poses)

# 사용 5 : 원문의 어절 정보를 같이 가져 오기
text_analyzed = rhinoMorph.wholeResult_text(rn, text)
print('\n5. 형태소 분석 결과:\n', text_analyzed)
'''