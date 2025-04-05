# SRTP
## 本地部署开发环境配置
### 1.用 Anaconda Prompt 创建一个虚拟环境
### 2.用 Anaconda Prompt 在虚拟环境中安装相关依赖
（1）安装PaddlePaddle框架（CPU版本）： pip install paddlepaddle

（2）安装PaddleOCR whl包： pip install paddleocr

（3）安装openai whl包： pip install openai

（4）安装streamlit框架： pip install streamlit

（5）安装python-docx whl包： pip install python-docx

（6）安装pdfplumber whl包： pip install pdfplumber
### 3.配置Pycharm
在Pycharm界面的右下角有项目的python解释器设置，点击“添加新的解释器”，点击“Conda环境”，找到虚拟环境所对应的conda可执行文件——..\Anaconda3\Scripts\conda.exe，点击“加载环境”。
### 4.运行app_streamlit.py
在Pycharm界面的左下角找到“终端”，输入命令streamlit run app_streamlit.py，再回车，即可运行网页。