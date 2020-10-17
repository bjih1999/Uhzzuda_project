import os
import time
import csv

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv

load_dotenv(verbose=True)

driver = webdriver.Chrome('chromedriver_win32/chromedriver.exe')
driver.implicitly_wait(1)
driver.get('https://everytime.kr/login')

driver.find_element_by_name('userid').send_keys(os.getenv('everytime_id'))
driver.find_element_by_name('password').send_keys(os.getenv('everytime_pw'))
driver.find_element_by_xpath('//*[@id="container"]/form/p[3]/input').click()

driver.get('https://everytime.kr/lecture')

f = open('lecturelist.csv', 'r')
lecturelist = csv.reader(f)

reviewlist = []
for l in lecturelist:
    for ll in l:
        if ll == '':
            continue
        else:
            # 과목 검색
            elem = driver.find_element_by_name('keyword')
            elem.clear()
            elem.send_keys(ll)
            elem.submit()
            time.sleep(0.5)

            #강의 목록 하나씩 탭열기
            lectures = driver.find_elements_by_class_name('lecture')
            for lec in lectures:
                print("-------------------------")

                lec.send_keys(Keys.CONTROL + "\n")
                driver.switch_to.window(driver.window_handles[-1])

                #강의평 하나씩 출력해봄
                try:
                    articles = driver.find_element_by_class_name("articles")
                    reviews = articles.find_elements_by_class_name("text")
                    #cnt = 1
                    for review in reviews:
                       # print(cnt)
                        #cnt += 1
                        print(review.text)
                        reviewlist.append(review.text)
                except:
                    pass
                driver.close()
                driver.switch_to.window(driver.window_handles[0])


with open('reviewlist.csv', 'w', newline='') as reviewfile:
    writer = csv.writer(reviewfile)
    writer.writerow(reviewlist)
