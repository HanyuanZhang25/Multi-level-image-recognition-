# Multi-level-image-recognition-
This project is a multi-level image recognition computer vision model used for the storage area of gas stations. By training the multi-level image recognition model and combining with scanning technology, real-time material recognition has been achieved. The entire system will classify the items to be entered into the warehouse into 6 categories. These 6 categories of items will be identified by the system and separately counted and classified during the entry process. The 6 categories of items are: fueling gun, pipe joint, IBC tanker filling equipment, urea pump, oil outlet pipe, and oil outlet pipe valve. Among them, except for these six categories of items that need to be classified separately, the pipe joint and the oil outlet pipe valve need to be classified again. Moreover, after the multi-level image recognition model is constructed, we need to enable users to update and optimize the model based on the original network and new data.

The project analysis is as follows:
1： class1_code: Used for the internal classification of the pipe joint
（1）	The test and train data folders contain the training and evaluation data for training and evaluation
（2）	api1 is the interface for the model class1, facilitating the calling by front-end personnel
（3）	best_class1.pth: Stores the coefficients after training the model
（4）	class_map and class_map1.Json are dictionary documents that correspond to our specified classifications. The former is the overall classification of type 1 items, and the latter is the internal classification of class1 pipe joints
（5）	train_test_class1.py: Internal classification model for pipe joints, both for training and testing (to switch to the testing mode, please refer to the comments in the main function of this model)
2:   class5_code: This is the internal classification for the oil pipe valve. 
The basic structure is the same as class1_code.
3:   new_train_model: This contains the file "fine_tune_improve.py", which is used to enable users to update and optimize the model based on the original network and additional data after the multi-level image recognition model has been constructed.
4:   total_class_code: Stores 6 types of total classifications
(1)  app: Integrates 6 types of total classification models and 2 sub-type internal classification models, and designs an entry point for easy front-end invocation
(2)  train_test: 6 classification models, both trainable and testable. To switch to test mode, please refer to the comments in the main function of this model
(3)  best_prototypical_encoder: Parameters after training the model
(4)  get_data: Used for data augmentation, rotates and symmetrizes the original data to increase the size of the dataset
(5)  gui: Generates a small graphical user interface to simulate connection with the front-end

5: total_data:  Stores data, namely the internal data of class1, the internal classification data of class5, the total classification data of total_data (original data, without data augmentation), and the total classification data of total_data (after data augmentation by get_data)

该项目是一款用于加油站存储区域的多级图像识别的计算机视觉模型。通过训练多级图像识别模型并结合扫描技术，实现了实时的物料识别。整个系统将需要入库的物品分为6类，这6类物品将被系统识别并在入库时费别统计并分类。6类物品分别是：加油枪、管接头、IBC吨箱加注设备、尿素泵、出油管以及出油管阀。其中除了这六类物品需要分别分类外，管接头和和出油管阀门需要再度分类。而且，在多级图像识别模型构建完成后，我们需要使得使用者能够基于原始网络和新增数据对模型进行更新优化
项目分析如下：
1：class1_code:用于管接头的内部分类
（1）	test数据和train数据文件夹含有训练和评估的数据，用于训练和评估
（2）	api1为模型class1.的接口，方便前端人员调用
（3）	best_class1.pth：存储训练后的系数
（4）	class_map和class_map1.Json是我们指定的分类对应的一个字典文档，前者为1类物品的总分类，后者是class1管接头的内部分类
（5）	train_test_class1.py：管接头内部分类模型，可训练也可测试，（要转换为测试模式，详见该模型main中的注释）
2：class5_code:用于出油管阀的内部分类
	基本结构同class1_code
3：new_train_model：内含文件fine_tune_improve.py，用于在多级图像识别模型构建完成后，使得使用者能够基于原始网络和新增数据对模型进行更新优化。
4：total_class_code：存储了6个类型的总分类
	（1） app：集成了六个类型的总分类模型和两个子类型的内部分类模型，并设计了启动入口方便前端调用
  （2） train_test：6分类模型，可训练也可测试，要转换为测试模式，详见该模型main中的注释）
	（3） best_prototypical_encoder：训练完成后的参数
	（4） get_data:用于数据增强，将原数据进行旋转，对称等处理，提高数据集大小
	（5） gui：生成一个小的图形用户界面，方便与前端对接
5：total_data：存储数据，分别为class1内部数据，class5内部分类数据，total_data总分类数据（原数据，未经数据增强），total_data总分类数据（经过get_data的数据增强）


