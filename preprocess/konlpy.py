from konlpy.tag import Hannanum
from pykospacing import spacing
import csv
import re

hannanum = Hannanum()

reviewlist = []
sentences = []
with open('../reviewlist/reviewlist.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)

    for review in reviewlist:
        review = review[0]
        review_entered = review.split('\n')

        for review_splitted in review_entered:
            sentences.append(review_splitted)

reviewlist = sentences
with open('./jjin_pre_review.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    preprocessed_sent = []
    for review in reviewlist:
        review = review[0]
        review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)  # [20nn-n학기 ~~대학 공모전 ~]과 같은 형식 제거
        review = re.sub(r'(\*)([\w\s\/,]*)(.)', '', review)  # ex) *본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다. 와 같은 형식 제거
        # review = re.sub(r'(\*)([\w\s:]*)(★*)', '', review) # ex) * 공과대학 학우들에게 추천하는 정도:★★★★★ 제거
        review = re.sub(r'(-)([\w\s-]*)(.)', '',
                        review)  # ex) - 공과대학 학우분들이 답변해 주신 솔직한 후기로 모든 후기들은 에브리타임 수강평 등록을 허락하신 학우님들에 한하여 작성되었습니다. 제거
        review = re.sub(r'(\*)([\w\s\/,]*)(\**)', '',
                        review)  # ex) ** 20-1학기 IT대학 강의평가 공모전을 통해 접수된 강의평가입니다. ** 와 같은 형식 제거
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
        review = review.replace('에쁠', 'A+')
        review = review.replace('에이쁠', 'A+')
        review = review.replace('에이플', 'A+')
        review = review.replace('에이제로', 'A0')
        review = review.replace('에이마', 'A-')
        review = review.replace('에이', 'A')
        review = review.replace('비쁠', 'B+')
        review = review.replace('비플', 'B+')
        review = review.replace('씨쁠', 'C+')
        review = review.replace('씨플', 'C+')
        review = review.replace('에이쁠', 'A+')
        review = review.replace('고사', '')
        review = re.sub(r'[ㄱ-ㅎ]', '', review)  # 자음제거
        review = re.sub(r'[ㅏ-ㅣ]', '', review)  # 모음제거
        review = review.upper()
        spaced_review = spacing(review)
        morphs = hannanum.pos(spaced_review, ntags=22)
        print(morphs)
        sent=[]
        for morph in morphs: # morph[1].startswith('M') == False and
            if  morph[1] != 'II' and morph[1].startswith('J') == False and morph[1].startswith('S') == False and morph[1].startswith('E') == False:
                if morph[0] == '강':
                    sent.append('강')
                else:
                    sent.append(morph[0])

            if morph[1].startswith('S') or (morph[1].startswith('E') and '고' in morph[0]) or (morph[1] == 'ET' and '음' in morph[0] or '임' in morph[0] or 'ㅁ' in morph[0]) or morph[1] == 'EF':
                if len(sent) != 0:
                    preprocessed_sent.append(sent)
                    print(sent)
                    writer.writerow(sent)
                    sent = []

        if len(sent) != 0:
            preprocessed_sent.append(sent)
            print(sent)
            writer.writerow(sent)