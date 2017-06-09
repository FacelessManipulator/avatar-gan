#!/bin/bash
python3 getUrl.py 1 100 --worker=4
paste -s -d \\n bangumiImgUrl-0.csv bangumiImgUrl-1.csv bangumiImgUrl-2.csv bangumiImgUrl-3.csv > bangumiImgUrl.csv
rm -f bangumiImgUrl-*.csv
python getImg.py bangumiImgUrl.csv 1 --worker=4
