from konlpy.tag import Hannanum
from pykospacing import spacing
import csv
import re

hannanum = Hannanum()

reviewlist = []
with open('konlpy/reviewlist2.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist.append(row)

# print(reviewlist[:10])
#
# spaced_sent = spacing("수업도 괜찮았고 교수님 굉장히 친절하셨고 세세하게 잘 알려주셨습니다. 시험 난이도도 적당했습니다. 추천드립니다. 부담없는 내용, 하지만 팀플은 너무 싫었음 외부강연도 많고 지루하지는 않았었던 것 같음")
# print(sp)
# x1 = hannanum.pos(x)
# preprocessed_sent=[]

# regex1 = re.compile(r'(^\[)($])')
# print(re.sub(r'^\[[\w\b]+\]$', '', '[2020-1학기 공과대학 수강후기 공모전]'))
# print(re.sub(r'(\[)([\w\s-]*)(\])', '', '[2018-2 경영대학 리얼 수강후기 공모전]\n*상단의 과제/조모임/학점비율 등의 항목들은 일괄적으로 가장 왼쪽 항목에 체크되었으며, 아래의 본 내용과 무관함을 알려드립니다.\n*본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다.\n교수님의 수업 스타일은 매우 자유분방 하십니다. 너무 자유분방하셔서 수업의 본질을 흐리시는 것 같네요 금융시장론이란 명칭을 바꾸셔야 될거 같습니다. 밴드라는 걸 이용해서 매주 기사 스크랩을 시키고 참잘했어요를 주십니다 시험은 기말고사 한번인데 비중이 매우 적습니다 출결은 거의 반영 안되는 것 같구요 성적 비중을 교수님 마음대로 정합니다 팀프로젝트가 한 번 있어요').replace('*상단의 과제/조모임/학점비율 등의 항목들은 일괄적으로 가장 왼쪽 항목에 체크되었으며, 아래의 본 내용과 무관함을 알려드립니다.', '').replace('*본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다.', ''))
with open('konlpy/preprocessed_review2.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    preprocessed_sent = []
    for review in reviewlist:
        print(review[0])
        review = review[0]
        review = re.sub(r'(\[)([\w\s-]*)(\])', '', review)
        review = re.sub(r'[~!@#$%^&*()♥★☆♡-]', '', review)
        review = review.replace('*상단의 과제/조모임/학점비율 등의 항목들은 일괄적으로 가장 왼쪽 항목에 체크되었으며, 아래의 본 내용과 무관함을 알려드립니다.', '')
        review = review.replace('*본 후기는 온전히 경영대학 학우 분들의 의견으로 구성되었습니다.', '')
        # review = review.replace('\"', '')
        # review = review.replace('♥', '')
        # review = review.replace('~', '')
        # review = review.replace('!', '')
        # review = review.replace('ㅋ', '')
        # review = review.replace('ㅠ', '')
        # review = review.replace('ㅜ', '')
        spaced_review = spacing(review)
        morphs = hannanum.pos(spaced_review)
        sent=[]
        for morph in morphs:
            if morph[1] != 'J' and morph[1] != 'E' and morph[1] != 'S':
                sent.append(morph[0])
        preprocessed_sent.append(sent)
        print(sent)
        writer.writerow(sent)


# print(preprocessed_sent)

