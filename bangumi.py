# -*- coding: utf-8 -*-
import scrapy
from scrapy import Request


class BangumiSpider(scrapy.Spider):
    name = "bangumi"
    allowed_domains = ["bangumi.tv"]
    start_urls = ['http://bangumi.tv/character?gender=2']
    f = open('ch.txt','wb')
    def parse(self, response):
        names = response.xpath("//div[@class='light_odd clearit']//h2//text()").extract()
        imgs = response.xpath("//div[@class='light_odd clearit']/a[1]/img/@src").extract()
        for person in zip(names, imgs):
            buf = "%s,%s"%(person[0], 'http:'+person[1])
            self.f.write(buf.encode('gb2312'))
        nextPageNum = int(response.meta['pageNum']) + 1 if hasattr(response.meta, 'pageNum') else 2
        yield Request(url='http://bangumi.tv/character?gender=2&page=%d'%nextPageNum,
                      callback=self.parse,
                      meta={'pageNum': str(nextPageNum)})
