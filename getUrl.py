# -*- coding: utf-8 -*-
import urllib
from urllib import request
import lxml.html.soupparser as soupparser
import multiprocessing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('startPage', type=int, help='The start page number, default 1', default=1)
parser.add_argument('endPage', type=int, help='The end page number, default 100', default=100)
parser.add_argument('--worker', type=int, help='The worker process', default=2)

opt = parser.parse_args()

class BangumiSpider(object):
    def __init__(self, startPage, endPage, threadsCount=1):
        self.urlTemp = 'http://bangumi.tv/character?gender=2&page={0}'
        self.csvPath = 'bangumiImgUrl-{0}.csv'
        self.startIndex = startPage
        self.endIndex = endPage
        self.threadsCount = threadsCount
        
    def parse(self, html_content, csvFile):
        html = soupparser.fromstring(html_content)
        names = html.xpath("//div[@class='light_odd clearit']//h2//text()")
        imgs = html.xpath("//div[@class='light_odd clearit']/a[1]/img/@src")
        for character in zip(names, imgs):
            csvFile.write("%s,%s\n"%(character[0].strip(), 'http:'+character[1]))
        return len(names)
            
    def request(self, urls, parser, id):
        count = 0
        with open(self.csvPath.format(id), 'w') as f:
            for url in urls:
                try:
                    data = request.urlopen(url).read()
                    count += parser(data.decode('utf-8'), f)
                except RuntimeError as e:
                    print('%s parsing failed.'%url)
        print("Process %d done. %d url parsed."%(id, count))
                
    def run(self):
        processes = []
        urls = [self.urlTemp.format(index) for index in range(self.startIndex, self.endIndex+1)]
        for i in range(self.threadsCount):
            url = urls[i::self.threadsCount]
            process = multiprocessing.Process(target = self.request, args = (url, self.parse, i))
            process.start()
            processes.append(process)
        print("Running spider with %d processes"%self.threadsCount)
        return processes

if __name__ == '__main__':
    bangumiSpider = BangumiSpider(opt.startPage, opt.endPage, opt.worker)
    processes = bangumiSpider.run()
#    map(lambda p:p.join(), processes)

