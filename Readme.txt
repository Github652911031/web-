web搜索大作业说明：
1.EM_extend.py文件实现了扩展的EM算法
2.sml.py文件实现图片数据导入，训练每张图片的8个高斯混合模型。同时导入EM_extend.py文件中的扩展EM模块，完成整个大作业的流程。
3.在执行sml.py文件时，确保目录下含有model_picture和model_label_raw两个文件夹用于保存训练过程中的数据。
4.64个协方差矩阵保存在.npz文件中，只取了label0的数据。
