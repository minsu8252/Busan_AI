from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import urllib.request 
import time

# def hasxpath(xpath):
#     try:
#         driver.find_element_by_xpath(xpath)
#         return True
#     except:
#         return False

# 웹페이지 열기
path = "chromedriver.exe"
driver = webdriver.Chrome(path)
base_url = "https://store.musinsa.com/app/styles/lists?use_yn_360=&style_type=&brand=&model=&tag_no=&max_rt=&min_rt=&display_cnt=60&list_kind=big&sort=date&page="
cnt = 1
for page in range(1, 3): # 페이지마다 크롤링을 위해 반복문 사용 / range를 변경하여 이미지저장 개수를 조절 가능
    url = base_url + str(page)
    driver.get(url)

    time.sleep(1)  # 페이지 로딩을 위해 약간의 텀을 둠

    # # 썸네일 이미지 저장
    # # 페이지 내에서 이미지 저장
    # images =driver.find_elements_by_css_selector(".style-list-thumbnail__img")
    # for image in images:
    #     imgUrl = image.get_attribute("src")
    #     urllib.request.urlretrieve(imgUrl, "musinsa_img_data/" + str(cnt) + ".jpg") # 저장할 위치 및 저장이름 / 미리 저장할 위치(폴더)를 만들어야함
    #     cnt += 1

    # 큰 사진 저장 / 120장 중에서 75장만 크롤링 된다.
    # images =driver.find_elements_by_css_selector(".style-list-thumbnail__img")
    for i in range(60):
        try:
            images =driver.find_elements_by_css_selector(".style-list-thumbnail__img")
            image = images[i]
            image.click()
            time.sleep(2)
            imgUrl = driver.find_element_by_xpath("/html/body/div[3]/div[3]/div[3]/div[2]/div[2]/div/div/img[2]").get_attribute("src")
        
            
            # if hasxpath("/html/body/div[3]/div[3]/div[3]/div[2]/div[2]/div/div/video[1]") == True:
            #     imgUrl = driver.find_element_by_xpath("/html/body/div[3]/div[3]/div[3]/div[2]/div[2]/div/div/img[2]").get_attribute("src")
                    
            # else: 
            #     imgUrl = driver.find_element_by_xpath("/html/body/div[3]/div[3]/div[3]/div[2]/div[2]/div/div/img[2]").get_attribute("src")

            urllib.request.urlretrieve(imgUrl, "musinsa_big_img_data/" + str(cnt) + ".jpg") # 저장할 위치 및 저장이름 / 미리 저장할 위치(폴더)를 만들어야함
            cnt += 1
            driver.back()
        except:
            pass
        

# 웹 닫기
driver.close()