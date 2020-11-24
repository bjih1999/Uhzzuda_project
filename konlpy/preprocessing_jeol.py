from konlpy.tag import Hannanum
from pykospacing import spacing
import csv
import re

hannanum = Hannanum()

reviewlist = []
sentences = []
with open('konlpy/reviewlist.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)

    for review in reviewlist:
        review = review[0]
        review_entered = review.split('\n')

        for review_splitted in review_entered:
            sentences.append(review_splitted)

reviewlist = sentences

# print(reviewlist[:10])
#
# spaced_sent = spacing("수업도 괜찮았고 교수님 굉장히 친절하셨고 세세하게 잘 알려주셨습니다. 시험 난이도도 적당했습니다. 추천드립니다. 부담없는 내용, 하지만 팀플은 너무 싫었음 외부강연도 많고 지루하지는 않았었던 것 같음")
# print(sp)
# x1 = hannanum.pos(x)
# preprocessed_sent=[]

# regex1 = re.compile(r'(^\[)($])')
# print(re.sub(r'^\[[\w\b]+\]$', '', '[2020-1학기 공과대학 수강후기 공모전]'))
# print(re.sub(r'(\[)([\w\s-]*)(\])', '', '[2018-2 경영대학 리얼 수강후기 공모전]\n*상단의 과제/조모임/학점비율 등의 항목들은 일괄적으로 가장 왼쪽 항목에 체크되었으며, 아래의 본 내용과 무관함을 알려드립니다.\n*본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다.\n교수님의 수업 스타일은 매우 자유분방 하십니다. 너무 자유분방하셔서 수업의 본질을 흐리시는 것 같네요 금융시장론이란 명칭을 바꾸셔야 될거 같습니다. 밴드라는 걸 이용해서 매주 기사 스크랩을 시키고 참잘했어요를 주십니다 시험은 기말고사 한번인데 비중이 매우 적습니다 출결은 거의 반영 안되는 것 같구요 성적 비중을 교수님 마음대로 정합니다 팀프로젝트가 한 번 있어요').replace('*상단의 과제/조모임/학점비율 등의 항목들은 일괄적으로 가장 왼쪽 항목에 체크되었으며, 아래의 본 내용과 무관함을 알려드립니다.', '').replace('*본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다.', ''))
with open('konlpy/preprocessed_review_jeol_test.csv', 'w', newline='') as output_file:
    #test 중
    writer = csv.writer(output_file)
    preprocessed_sent = []
    for review in reviewlist[:110]:
        print(review)
        # review = review[0]
        review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)  # [20nn-n학기 ~~대학 공모전 ~]과 같은 형식 제거
        review = re.sub(r'(\*+)([\w\s\.\-]*)(\*+)', '', review)  # ex) ** 20-1학기 IT대학 강의평가 공모전을 통해 접수된 강의평가입니다. ** 와 같은 형식 제거
        review = re.sub(r'(\*)([\w\s\/,]*)(.)', '', review)  # ex) *본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다. 와 같은 형식 제거
        # review = re.sub(r'(\*)([\w\s:]*)(★*)', '', review) # ex) * 공과대학 학우들에게 추천하는 정도:★★★★★ 제거
        review = re.sub(r'(-)([\w\s-]*)(.)', '', review) # ex) - 공과대학 학우분들이 답변해 주신 솔직한 후기로 모든 후기들은 에브리타임 수강평 등록을 허락하신 학우님들에 한하여 작성되었습니다. 제거
        review = re.sub(r'[~!@#$%^&*()♥★☆♡-]', '', review)  # 특수문자 제거
        review = review.replace('ㅋ', '')
        # review = review.replace('ㅎ', '')
        review = review.replace('ㅜ', '')
        review = review.replace('ㅠ', '')
        review = review.replace('"', '')
        spaced_review = spacing(review)
        morphs = hannanum.pos(spaced_review, ntags=22)
        print(morphs)
        sent=[]
        for morph in morphs: # morph[1].startswith('M') == False and
            if  morph[1] != 'II' and morph[1].startswith('J') == False and morph[1].startswith('S') == False and morph[1].startswith('E') == False:
                if morph[0] == '강':
                    sent.append('강의')
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
