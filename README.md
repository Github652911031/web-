# web-
web搜索大作业:利用GMM进行图片标注
主要步骤：1.每张EM算法计算出8个GMM模型
2.每类图片用扩展的EM算法计算出64个GMM模型
data文件夹存放图片数据
model_picture为训练好的数据模型

data_process.py 数据预处理，计算每张图片的GMM参数
EM_extend.py 扩展的EM模型
train.py 对每类图片训练
sml.py是整合文件，整合以上三个文件
