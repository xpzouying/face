# 人脸识别比对平台 #
人脸识别比对平台


# 简介 #
本项目通过TensorFlow搭建一套人脸识别平台，提供人脸相关的Web Service。  

提供：
- [x] 比对/verify
- [ ]人脸检测并截取
- [ ] VIP识别
- [ ] 告警


本文源码链接为：[https://github.com/xpzouying/face](https://github.com/xpzouying/face)

目前版本处于初级阶段，有任何问题可以联系我进行交流。
> email: xpzouying@gmail.com


# 目标 #
提供人脸比对的网络服务，通过访问该人脸识别服务，返回比对结果，判断比对的是不是同一个人。
> ```bash
> # Run web service
> python main.py
>
> # Post request for verify
> curl -X POST -F "img1=@1.jpg" -F "img2=@2.jpg" http://localhost:8080/verify
> ```

两张图片
> 1.jpg  
> ![1.jpg](/images/zy/verify_compare_1.jpg)

> 2.jpg:  
> ![2.jpg](/images/zy/verify_compare_2.jpg)


比对结果
> ```json
> {
>   "is_same_person": "False",
>   "time_used": "1.81232595444"
> }
> ```
> is_same_person: 判断是不是一个人  
> time_used: 比对耗时，单位秒(s)


# 技术选型 #
- Facenet / Tensorflow
- Flask
    > 使用Flask框架搭建Web Service