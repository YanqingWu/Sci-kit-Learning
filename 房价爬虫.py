
#本爬虫关键bug在 def insert 函数，insert into 到 value 那部分特别容易出现格式错误，要带到 navicat 里面反复试验!!!!!!!!

def get_house_data(url):
    import pymysql

    import requests 

    from bs4 import BeautifulSoup


    # url = 'https://nj.lianjia.com/ditiefang/li110460689/'  # test url


    def get_pageinfo(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text)
        return soup

    def get_links(url):
        soup = get_pageinfo(url)
        links_info = soup.find_all('li',class_ = 'clear LOGCLICKDATA')
        links = [li.a.get('href') for li in links_info]
        return links

    # house_url = 'https://nj.lianjia.com/ershoufang/103102286934.html'  #test house_url

    def get_house_info(house_url):
        soup = get_pageinfo(house_url)
        prince = soup.find('span',class_ = 'total').text
        prince_unit = soup.find('span',class_ = 'unit').text.strip()
        house_maininfo = soup.find_all('div',class_ = 'mainInfo')
        area = house_maininfo[2].text
        direction = house_maininfo[1].text
        layout = house_maininfo[0].text
        floor = soup.find('div',class_ = 'subInfo').text
        location = soup.find('span',class_ = 'info').text.strip().split('\xa0')
        location = ''.join(location)
        community = soup.find('div',class_ = 'communityName').a.text

        house_info = {
            '价格':prince,    
            '价格单位':prince_unit,
            '面积':area,
            '朝向':direction,
            '房型':layout,
            '楼层':floor,
            '区域':location,
            '小区':community,
                    }
        return house_info

    db = pymysql.connect(user = 'root',password = 'wyq517517',database = 'house',host = '127.0.0.1')

    def insert (db,house):

        values = "'{}'," * 7 + "'{}'" 
        sql_values = values.format(house['价格'],house['价格单位'],house['面积'],house['朝向'],house['房型'],house['楼层'],house['区域'],house['小区']) 

        sql = """insert into house(prince,prince_unit,area,direction,layout,floor,location,community) values ({})
        """.format(sql_values)

        cursor = db.cursor()
        print(sql)
        cursor.execute(sql)
        db.commit()

    links = get_links(url)
    for house_url in links:
        house = get_house_info(house_url)
        insert(db,house)


url = input('请输入链家网址： ')
get_house_data(url)

#利用pandas把导入的数据写入pandas.dataframe

import pandas as pd 
import pymysql    
con = pymysql.connect(user = 'root',password = 'wyq517517',database = 'house',host = '127.0.0.1') 
sql_cmd = "SELECT * FROM house"
data = pd.read_sql(sql_cmd, con)