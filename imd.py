import pandas
import numpy
from selenium.webdriver.common.keys import Keys
import sys
sys.path.append('C:\\web_driver')
from web_driver import *
import time
# nltk.download()

text = pandas.read_csv('C:\\netflix\\netflix_titles.csv',engine = 'python',sep = ',')
text = text.apply(lambda x: x.apply(lambda y: str(y).lower()))
text['note'] = ''

wd = web_driver_selenium()
wd.create_browser()
wd.get_to_page('https://www.imdb.com/')
start = time.time()
for film in range(text.shape[0]):
    print(str(film) + ' - ' + str(text.shape[0]))
    try:
        element = wd.find_element('//input[@type = "text"]')
        wd.fill_form(element,text['title'].iloc[film])
        wd.fill_form(element,Keys.RETURN)
        done_q = True
    except:
        pass
    try:
        start_element = time.time()
        catch_element = False
        while time.time() - start_element <= 10 and catch_element == False:
            try:
                element = wd.find_element('//td[@class ="result_text"]//a')
                catch_element = True
            except:
                pass
        url_movie = wd.find_attribute(element,'href')
        wd.get_to_page(url_movie)
        start_element = time.time()
        catch_element = False
        while time.time() - start_element <= 10 and catch_element == False:
            try:
                element = wd.find_element('//div[@class="imdbRating"]//strong')
                catch_element = True
            except:
                pass
        text['note'].iloc[film] = wd.find_attribute(element,'title')
    except:
        wd.get_to_page('https://www.imdb.com/')
    if time.time() - start > 300:
        wd.close_browser()
        wd = web_driver_selenium()
        wd.create_browser()
        wd.get_to_page('https://www.imdb.com/')
        start = time.time()
try:
    wd.close_browser()
except:
    pass
text.to_csv('C:\\netflix\\netflix_titles_imdb.csv',sep = ';',index = False)
