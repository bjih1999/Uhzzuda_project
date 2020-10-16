import os
import time
import csv

from selenium import webdriver
from dotenv import load_dotenv

load_dotenv(verbose=True)

driver = webdriver.Chrome('chromedriver_win32/chromedriver.exe')
driver.implicitly_wait(1)
driver.get('https://everytime.kr/login')

driver.find_element_by_name('userid').send_keys(os.getenv('everytime_id'))
driver.find_element_by_name('password').send_keys(os.getenv('everytime_pw'))
driver.find_element_by_xpath('//*[@id="container"]/form/p[3]/input').click()

driver.get('https://everytime.kr/timetable/2018/1')
driver.find_element_by_xpath('//*[@id="container"]/ul/li[1]').click()
driver.implicitly_wait(3)

SCROLL_PAUSE_TIME = 5

last_lecture = ''
while True:
    lectures = driver.find_elements_by_css_selector("td.bold")
    # Scroll down to bottom
    last = lectures[-1]
    driver.execute_script("arguments[0].scrollIntoView(true);", last)

    time.sleep(SCROLL_PAUSE_TIME)

    last_lecture = driver.find_elements_by_css_selector("td.bold")[-1]

    if last == last_lecture:
        break

lectureList = []
for l in lectures:
    lectureList.append(l.text)

driver.get('https://everytime.kr/timetable/2018/2')
driver.find_element_by_xpath('//*[@id="container"]/ul/li[1]').click()
driver.implicitly_wait(3)

last_lecture = ''
while True:
    lectures = driver.find_elements_by_css_selector("td.bold")
    # Scroll down to bottom
    last = lectures[-1]
    driver.execute_script("arguments[0].scrollIntoView(true);", last)

    time.sleep(SCROLL_PAUSE_TIME)

    last_lecture = driver.find_elements_by_css_selector("td.bold")[-1]

    if last == last_lecture:
        break

for l in lectures:
    lectureList.append(l.text)

myset = set(lectureList)
mylist = list(myset)

with open('lecturelist.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(mylist)