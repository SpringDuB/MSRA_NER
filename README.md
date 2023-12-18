# MSRA_NER
基Seq2Seq的命名实体识别

 使用Bert-base-chinese 在新闻数据集上训练，实体有LOC，PER，ORG，数据集采用B-I-O的前缀，如B-LOC表示地点实体的开始， O表示不是实体， I-LOC表示实体中

F1值最好在92

开始训练

`python training.py --fine_tune_root_dir bert模型文件夹 --data_path 数据集文件夹`

模型部署：

`python server_running.py` 

默认使用output/model/ckpt/中的模型，可更改
