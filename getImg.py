import csv
import os
import hashlib
import multiprocessing
import urllib
from urllib import request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', type=str, help='csv file path')
parser.add_argument('column', type=int, help='The column which contains url', default=1)
parser.add_argument('--worker', type=int, help='The count of worker process', default=2)
parser.add_argument('--folder', type=str, help='folder to save images', default='./data')
opt = parser.parse_args()
class ImageDownloader(object):
    def __init__(self, urls, threadsCount=1, folder='./'):
        self.folder = folder
        self.threadsCount = threadsCount
        self.urls = urls
            
    def request(self, urls, id):
        count = 0
        for url in urls:
            try:
                imgSuffix = url.split('.')[-1]
                host = url.split('/')[2]
                imgPath = os.path.join(self.folder, hashlib.md5(url.encode('utf-8')).hexdigest()+'.%s'%imgSuffix)
                if os.path.isfile(imgPath):
                    continue
                else:
                    req = request.Request(url)
                    req.add_header('Host', host)
                    req.add_header('User-Agent', ' Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0')
                    res = request.urlopen(req).read()
                    with open(imgPath, 'wb') as img:
                        img.write(res)
                    count += 1
                    if count%100 == 0:
                        print("Process %d downloaded %d images."%(id, count))
            except Exception as e:
                print('%s download failed.'%url)
        print("Process %d done."%id)
                
    def run(self):
        for i in range(self.threadsCount):
            url = self.urls[i::self.threadsCount]
            multiprocessing.Process(target = self.request, args = (url, i)).start()
        print("Running image downloader with %d processes"%self.threadsCount)

if __name__ == '__main__':
    imgUrls = []
    if not os.path.isdir(opt.folder):
        os.makedirs(opt.folder)
    with open(opt.csvfile, 'r') as csvfile:
        urls = csv.reader(csvfile)
        for url in urls:
            imgUrls.append(url[opt.column])
#    imgDownloader = ImageDownloader(imgUrls,opt.worker,opt.folder)
    print('downloading images with %d processes'%opt.worker)
#    imgDownloader.run()
    print('error, ImgDownloader was not implemented.')
