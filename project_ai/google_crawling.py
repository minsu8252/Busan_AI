from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time # 이미지 로딩을 위해 텀을 줌
import urllib.request # url을 사용하여 이미지를 다운 받기위해 

# 웹페이지 열기
path = "chromedriver.exe"
driver = webdriver.Chrome(path)
url = "https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl"
driver.get(url)

# 웹에서 검색
elem = driver.find_element_by_name("q") # 검색창을 찾는다
elem.send_keys("화보") # 검색어 넣기
elem.send_keys(Keys.RETURN) # 엔터키

# 웹페이지 스크롤 내리기
SCROLL_PAUSE_TIME = 1

# 스크롤 높이 알기
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # 스크롤 내리기
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # 이미지 로딩시간 대기 / 오류가 나면 시간 조절하기
    time.sleep(SCROLL_PAUSE_TIME)

    # 새로운 스크롤 높이와 이전 스크롤 높이 비교
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element_by_css_selector(".mye4qd").click() # 결과 더보기버튼 누르기
        except:
            break
    last_height = new_height

# 이미지 선택 후 다운
images = driver.find_elements_by_css_selector(".rg_i.Q4LuWd") # 구글 작은 이미지를 리스트로 받기
count = 1
for image in images:
    try: # 오류 발생 시 넘어가기 위해 try, except을 사용
        image.click() # 작은 이미지 클릭
        time.sleep(2) # 이미지 로딩하기 위해 기다리게함
        imgUrl = driver.find_element_by_xpath("/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img").get_attribute("src")  # 이미지url 가져오기
        urllib.request.urlretrieve(imgUrl, "img_data/" + str(count) + ".jpg") # urlretrieve(url, PATH)
        count += 1
    except:
        pass

# 웹 닫기
driver.close()