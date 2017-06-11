# Crawler for Bangumi
Bangumi的Character里面的头像数据很不错，可惜他的[爬虫协议](http://bangumi.tv/robots.txt)不让爬/pic/
所以分成了两部分，getUrl可以解析图片URL，getImg为了遵守协议，就不实现了。
采用url手动收集了一个二次元头像数据集（混入了一些三次元），因为手有点酸只收集了1000多张  
# DCGAN with keras
尝试用Keras实现了一下DCGAN，Loss_D一如既往地远小于Loss_G，正好可以跟下面的WGAN对比。  
Keras 2.0.2的同学跑的时候可能会出现下面Error，猜测是Keras旧版本中融合模型的数据传递有损坏，将Keras升级为2.0.4就解决了
'''
InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'dense_4_input' with dtype float
	 [[Node: dense_4_input = Placeholder[dtype=DT_FLOAT, shape=[], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
'''
# DCGAN with pytorch
从pytorch-example里面抄过来的，对比Keras有很多可以借鉴的地方。
