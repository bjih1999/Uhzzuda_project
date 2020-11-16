import csv
import os
import time

from selenium import webdriver
from dotenv import load_dotenv

file = open('lecturelist.csv', 'r')
reader = csv.reader(file)

lectures=[]
for line in reader:
    lectures.append(line)

lectures = lectures[0]

load_dotenv(verbose=True)

driver = webdriver.Chrome('chromedriver_win32/chromedriver.exe')
driver.implicitly_wait(1)
driver.get('https://everytime.kr/login')

driver.find_element_by_name('userid').send_keys(os.getenv('everytime_id'))
driver.find_element_by_name('password').send_keys(os.getenv('everytime_pw'))
driver.find_element_by_xpath('//*[@id="container"]/form/p[3]/input').click()

driver.get('https://everytime.kr/lecture')
for lecture in lectures:
    # print(lecture)
    try:
        driver.find_element_by_class_name('keyword').send_keys(lecture)
        driver.find_element_by_xpath('//*[@id="container"]/form/input[2]').click()
        classes = driver.find_elements_by_class_name('lecture')
        print(classes)
        for _class in classes:
            print(_class.text)
            _class.click()
            time.sleep(1)
            driver.back()
        driver.find_element_by_class_name('keyword').clear()
    except:
        pass