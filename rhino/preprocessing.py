import re

import rhinoMorph
from pykospacing import spacing

def preprocess(review):
    rn = rhinoMorph.startRhino()
    preprocessed_sent = []
    preprocessed_sent_str = []
    review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)  # [20nn-n학기 ~~대학 공모전 ~]과 같은 형식 제거
    review = re.sub(r'(\*)([\w\s\/,]*)(.)', '', review)  # ex) *본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다. 와 같은 형식 제거
    review = re.sub(r'(-)([\w\s-]*)(.)', '', review)  # ex) - 공과대학 학우분들이 답변해 주신 솔직한 후기로 모든 후기들은 에브리타임 수강평 등록을 허락하신 학우님들에 한하여 작성되었습니다. 제거
    review = re.sub(r'(\*)([\w\s\/,]*)(\**)', '', review)  # ex) ** 20-1학기 IT대학 강의평가 공모전을 통해 접수된 강의평가입니다. ** 와 같은 형식 제거
    review = re.sub(r'(\[0-9])([\w\s-]*)(\])', '', review)
    review = re.sub(r'[~!@#$%^&*()♥★☆♡▽;:/"]', '', review)  # 특수문자 제거
    review = review.replace('ㅈㄴ', '')
    review = review.replace('존나', '')
    review = review.replace('졸라', '')
    review = review.replace('너무너무', '')
    review = review.replace('너무', '')
    review = review.replace('정말', '')
    review = review.replace('걍', '그냥')
    review = review.replace('과제', '숙제')
    review = review.replace('팀플', '팀프로젝트')
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
    review = review.replace('피피티', 'PPT')
    review = review.replace('쌉', '매우')
    review = review.replace('셤', '시험')
    review = review.replace('쁠러스', '+')
    review = review.replace('플러스', '+')
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
    review = review.replace('고사', '')
    review = review.replace('숙,제', ',숙제,')
    review = review.replace('기,말', ',기말,')
    review = review.replace('기,출', ',기출,')
    review = review.replace('기,대', ',기대,')
    review = review.replace('기,초', ',기초,')
    review = review.replace('알,채', ',안,채')
    review = review.replace('안채,워', ',안,채우')
    review = review.replace('재수,강', ',재수강,')
    review = review.replace('싸,강', ',싸강,')
    review = review.replace('강,의', ',강의,')
    review = review.replace('빡,세', ',빡세,')
    review = review.replace('빡,쎄', ',빡세,')
    review = review.replace('빡,셉', ',빡세,')
    review = review.replace('빡,셉', ',빡세,')
    review = review.replace('빡,치', ',빡치,')
    review = review.replace('빡,침', ',빡치,')
    review = review.replace('외,우', ',외우,')
    review = review.replace('외,울', ',외우,')
    review = review.replace('외,운', ',외우,')
    review = review.replace('전범,위', ',전체,범위,')
    review = review.replace('저,범위', ',전체,범위,')
    review = review.replace('오프,', ',오프라인,')
    review = review.replace('오프라,', ',오프라인,')
    review = review.replace('온오,프', ',온오프,')
    review = re.sub(r'[ㄱ-ㅎ]', '', review)  # 자음제거
    review = re.sub(r'[ㅏ-ㅣ]', '', review)  # 모음제거
    review = review.upper()

    spaced_review = spacing(review)
    result = rhinoMorph.wholeResult_list(rn, spaced_review)
    morphs = []
    for idx in range(len(result[0])):
        if result[1][idx] == 'NA' and (result[0][idx] == '플' or result[0][idx] == '쁠'):
            result[0][idx] = '+'
        morphs.append((result[0][idx].strip(), result[1][idx].strip()))
    #print(morphs)
    sent = []
    sent_str = []
    pos = False
    for morph in morphs:  # morph[1].startswith('M') == False and
        if morph[1] != 'VCP' and morph[1] != 'MAJ' and morph[1] != 'IC' and morph[1].startswith('J') == False and \
                morph[1] != 'EP' and morph[1] != 'EC' and morph[1] != 'EF' and morph[1] != 'ETM' and morph[1] != 'ETN' and morph[1] != 'SP' and morph[1] != 'SE' and morph[1] != 'SF' and morph[0] != '':
            sent.append(morph[0])
            if morph[1] != 'SF':
                sent_str.append(morph[0])

            if morph[1] != 'VCP' and morph[1] != 'MAJ' and morph[1] != 'IC' and morph[1].startswith('J') == False and \
                    morph[1] != 'EP' and morph[1] != 'EC' and morph[1] != 'ETM' and morph[1] != 'ETN' and morph[1] != 'SP' and morph[1] != 'SE' and morph[1] != 'SF' and morph[0] != '':
                pos = True
            if (morph[1] == 'ETN' and '음' in morph[0] or 'ㅁ' in morph[0]) or (
                    morph[1] == 'EF' and (morph[0].endswith('다') or morph[0].endswith('요'))) or morph[1] == 'SF' or \
                    morph[1] == 'SE':
                if len(sent) > 1 and pos == True:
                    preprocessed_sent_str.append(sent_str)
                    #print('sent : ', str(' '.join(sent_str)))
                    sent_str = []
                    pos = False

            if (morph[1] == 'ETN' and '음' in morph[0] or 'ㅁ' in morph[0]) or (
                    morph[1] == 'EF' and (morph[0].endswith('다') or morph[0].endswith('요'))) or morph[1] == 'SF' or \
                    morph[1] == 'SE':
                if len(sent) > 1:
                    preprocessed_sent.append(sent)
                    #print(sent)
                    sent = []

        if len(sent) > 1:
            preprocessed_sent.append(sent)
            #print(sent)

        if len(sent) > 1 and pos == True:
            preprocessed_sent_str.append(sent_str)
            #print('sent : ', str(' '.join(sent_str)))
            pos = False
    return sent