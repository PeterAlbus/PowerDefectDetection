# Power Defect Detection

# 电力设备缺陷检测模型平台

基于Flask的后端，基于Python 3.12

## Dependencies / 包含库

+ Flask
+ Flask-Cors
+ Flask-SQLAlchemy
+ SQLAlchemy

数据库使用sqlite，会在database文件夹下自动生成。不使用Git管理。

## Development  Standard / 开发规范

+ 采用Controller、Service、Dao三层架构开发
+ Controller层进行接口编写，存放和前端交互的消息处理。
+ Service层进行逻辑处理编写，存放数据处理逻辑及模型调用等。
+ Dao层进行数据库交互逻辑编写，存放和对应数据库的交互

## Recommended IDE Setup / 推荐IDE及插件

PyCharm

## Project Setup / 安装依赖

```sh
conda create -n env_name python=3.12   # 创建新的虚拟环境
source activate env_name      # 激活新建的虚拟环境
pip install -r requirements.txt
```

或

```
conda env create -f environment.yaml
```

