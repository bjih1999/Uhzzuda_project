import re
import rhinoMorph
from pykospacing import spacing

def preprocess(review):

    sentences = []
    review_entered = review.split('\n')
    for review_splitted in review_entered:
        sentences.append(review_splitted)

    preprocessed_sent = []
    rn = rhinoMorph.startRhino()
    for review in sentences:
        review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)  # [20nn-n학기 ~~대학 공모전 ~]과 같은 형식 제거
        review = re.sub(r'(\*)([\w\s\/,]*)(.)', '', review)  # ex) *본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다. 와 같은 형식 제거
        review = re.sub(r'(-)([\w\s-]*)(.)', '', review)  # ex) - 공과대학 학우분들이 답변해 주신 솔직한 후기로 모든 후기들은 에브리타임 수강평 등록을 허락하신 학우님들에 한하여 작성되었습니다. 제거
        review = re.sub(r'(\*)([\w\s\/,]*)(\**)', '', review)  # ex) ** 20-1학기 IT대학 강의평가 공모전을 통해 접수된 강의평가입니다. ** 와 같은 형식 제거
        review = re.sub(r'(\[0-9])([\w\s-]*)(\])', '', review)
        review = re.sub(r'[~!@#$%^&*()<>♥★☆♡▽;:/="\']', '', review)  # 특수문자 제거review = review.replace('[', '')
        review = review.replace(']', '')
        review = review.replace('ㅈㄴ', '')
        review = review.replace('존나', '')
        review = review.replace('졸라', '')
        review = review.replace('너무너무', '')
        review = review.replace('너무', '')
        ##
        review = review.replace('엄청', '')
        review = review.replace('되게', '')
        review = review.replace('매우', '')
        review = review.replace('상당히', '')
        review = review.replace('진짜', '')
        review = review.replace('굉장히', '')
        review = review.replace('완전', '')
        review = review.replace('아주', '')
        review = review.replace('겁나', '')
        review = review.replace('넘나', '')
        review = review.replace('꽤', '')
        review = review.replace('과제', '숙제')
        review = review.replace('레포트', '숙제')
        review = review.replace('보고서', '숙제')
        review = review.replace('플젝', '프로젝트')
        review = review.replace('전범위', '전체 범위')
        review = review.replace('조원', '팀원')
        review = review.replace('빡셈', '빡세')
        review = review.replace('빡쎄', '빡세')
        review = review.replace('빡쎄', '빡세')
        review = review.replace('셈', '세요')
        review = review.replace('결도', '결')
        review = review.replace('안채', '안 채')
        review = review.replace('다채', '다 채')
        review = review.replace('오프라인', '오프')
        review = review.replace('온오프', '온 오프')
        review = review.replace('수업', '강의')
        review = review.replace('조교', '교수')
        review = review.replace('성적', '학점')
        ##
        review = review.replace('정말', '')
        review = review.replace('걍', '그냥')
        review = review.replace('과제', '숙제')
        review = review.replace('팀플', '팀프로젝트')
        review = review.replace(',숙,제,', ',숙제,')

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
        review = re.sub(r',숙,제*.,', ',숙제,', review)
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
        # print(morphs)
        sent = []
        for morph in morphs:  # morph[1].startswith('M') == False and
            if morph[1] != 'VCP' and morph[1] != 'MAJ' and morph[1] != 'IC' and morph[1].startswith('J') == False and \
                    morph[1] != 'EP' and morph[1] != 'EC' and morph[1] != 'EF' and morph[1] != 'ETM' and morph[
                1] != 'ETN' and morph[1] != 'SP' and morph[1] != 'SE' and morph[1] != 'SF' and morph[0] != '':
                if morph[1] == 'VV' and morph[0] == '꾸':
                    sent.append('꿀')

                elif len(sent) > 0 and (sent[-1] == '기' and (
                        morph[0].startswith('말') or morph[0].startswith('출') or morph[0].startswith('대') or \
                        morph[0].startswith('초') or morph[0].startswith('분') or morph[0].startswith('억'))):
                    sent[-1] = '기' + morph[0][0]

                elif len(sent) > 0 and (sent[-1] == '숙' and morph[0].startswith('제') and morph[0].startswith('저')):
                    sent[-1] = '숙제'

                elif morph[1] == 'MAG' and morph[0] == '꽤' or morph[0] == '거의':
                    pass

                elif len(sent) > 0 and morph[0] == '강이':
                    sent[-1] += '강'

                elif len(sent) > 0 and (sent[-1] == '재수' and morph[0].startswith('강')):
                    sent[-1] = '재수강'
                    if len(morph[0]) > 1:
                        sent.append(morph[0][1:])
                else:
                    sent.append(morph[0])

            elif len(sent) > 1 and (sent[-1] == '알' and morph[0] == 'ㄴ'):
                sent.pop()
                sent.append('안')

            if (morph[1] == 'ETN' and '음' in morph[0] or 'ㅁ' in morph[0]) or (
                    morph[1] == 'EF' and (morph[0].endswith('다') or morph[0].endswith('요'))) or morph[1] == 'SF' or \
                    morph[1] == 'SE':
                if len(sent) > 1:
                    preprocessed_sent.append(sent)
                    # print(preprocessed_sent)
                    sent = []

        if len(sent) > 1:
            preprocessed_sent.append(sent)
            # sent = []
            # print(sent)

    dest_preprocessed_sent = []
    for temp_sent in preprocessed_sent:
        oneline = ','.join(temp_sent)
        oneline = ',' + oneline
        oneline = oneline.replace(',숙,제,', ',숙제,')
        oneline = oneline.replace(',셈,', ',세,')
        oneline = oneline.replace(',하셈,', ',하세,')
        oneline = oneline.replace(',하세,', ',하,')
        oneline = oneline.replace(',거의,', '')
        oneline = oneline.replace(',기,말,', ',기말,')
        oneline = oneline.replace(',기,출,', ',기출,')
        oneline = oneline.replace(',기,대,', ',기대,')
        oneline = oneline.replace(',기,초,', ',기초,')
        oneline = oneline.replace(',알,채,', ',안,채,')
        oneline = oneline.replace(',안채,워,', ',안,채우,')
        oneline = oneline.replace(',재수,강,', ',재수강,')
        oneline = oneline.replace(',강이,', ',강,')
        oneline = oneline.replace(',싸,강,', ',싸강,')
        oneline = oneline.replace(',인,강,', ',싸강,')
        oneline = oneline.replace(',연,강,', ',연강,')
        oneline = oneline.replace(',빡,세,', ',빡,세,')
        oneline = oneline.replace(',빡,쎄,', ',빡,세,')
        oneline = oneline.replace(',빡,셉,', ',빡,세,')
        oneline = oneline.replace(',빡,셈,', ',빡,세,')
        oneline = oneline.replace(',외,운,', ',외우,')
        oneline = oneline.replace(',전범,위,', ',전체,범위,')
        oneline = oneline.replace(',저,범위,', ',전체,범위,')
        oneline = oneline.replace(',빡,침,', ',빡,치,')
        oneline = oneline.replace(',외,우,', ',외우,')
        oneline = oneline.replace(',외,울,', ',외우,')
        oneline = oneline.replace(',오프,', ',오프라인,')
        oneline = oneline.replace(',오프라,', ',오프라인,')
        oneline = oneline.replace(',온오,프', ',온오프,')
        oneline = oneline.replace(',결도,', ',결,')
        dest_sent = oneline.replace('\n', '').split(',')
        dest_sent = list(filter(None, dest_sent))
        dest_preprocessed_sent.append(dest_sent)

    return dest_preprocessed_sent, len(dest_preprocessed_sent)
