from koalanlp import API
from koalanlp.proc import Tagger
from koalanlp.Util import initialize, finalize
import csv


initialize(java_options="-Xmx4g", rhino='LATEST')

reviewlist = []

with open('entered_review.csv', 'r') as reviewfile:
    reader = csv.reader(reviewfile)
    for row in reader:
        reviewlist = row


tagger = Tagger(API.RHINO)
with open('rhino_sentence.csv', 'w', newline='') as outputfile:
    writer = csv.writer(outputfile)
    for review in reviewlist:
        taggedLine = tagger(review)
        if not taggedLine:
            continue
        print("-----------------")
        print(f"===>{taggedLine}")
        sent = ""
        for line in taggedLine:
            for word in line:
                for morph in word:
                    # print(f"{morph.getSurface()} {morph.getTag()}")
                    # 종결어미, 명사형전성어미
                    if morph.getTag().startsWith("EF") or morph.getTag().startsWith("ETN"):
                        sent = sent.strip()
                        if morph.getSurface() != "기" and morph.getSurface() != "다기" and morph.getSurface() != "까" and morph.getSurface() != "라"\
                                and morph.getSurface() != "거라" and morph.getSurface() != "을까" and morph.getSurface() != "아"\
                                and morph.getSurface() != "게" and morph.getSurface() != "ㄹ게" and morph.getSurface() != "을게"\
                                and morph.getSurface() != "나" and morph.getSurface() != "려나" and morph.getSurface() != "구나":
                            sent += morph.getSurface()
                            writer.writerow([sent])
                            #print(f"1 {sent}")
                            sent = ""
                    # 접속부사 (그리고)
                    elif morph.getTag().startsWith("MAJ"):
                        sent = sent.strip()
                        if sent != "":
                            writer.writerow([sent])
                            #print(f"2 {sent}")
                            sent = ""
                            sent += morph.getSurface()
                    #. ? ! ...
                    elif morph.getTag().startsWith("SF") or morph.getTag().startsWith("SE"):
                        sent = sent.strip()
                        if sent != "":
                            writer.writerow([sent])
                            #print(f"3 {sent}")
                            sent = ""
                    else:
                        sent += morph.getSurface()

            if sent != "":
                sent = sent.strip()
                #print(f"4 {sent}")
                writer.writerow([sent])
                sent = ""

finalize()