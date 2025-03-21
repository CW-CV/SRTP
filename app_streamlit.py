import streamlit as st
import cv2
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from openai import OpenAI

# 标题
st.title("基于文心大模型的智能阅卷平台")
# 上传答题卡图片
answer_sheet_image = st.file_uploader(
    label="请上传答题卡图片（jpg或jpeg或png格式）",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)
if answer_sheet_image is not None:
    # 上传的文件不为空
    st.success("上传图片成功！")
    image = np.array(bytearray(answer_sheet_image.read()), dtype='uint8')  # 转为数组
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # 字节编码
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 图片通道转换
    st.image(RGB_img, channels="RGB") # 显示图片
    # 开始检测和识别按钮
    det_and_rec = st.button("点击开始检测和识别")
    # 使用会话状态
    if "det_and_rec" not in st.session_state:
        st.session_state.det_and_rec = det_and_rec
    else:
        st.session_state.det_and_rec = True
    # 按钮按下
    if st.session_state.det_and_rec:
        ocr = PaddleOCR(rec_model_dir='/infer')  # 设置自己训练的模型的路径，模型路径下必须含有model和params文件
        img_path = 'test3.jpg'  # 设置测试图片的路径
        result = ocr.ocr(img_path, cls=True)  # 进行文本检测、方向分类和文本识别

        result = result[0]
        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]  # 检测框
        txts = [line[1][0] for line in result]  # 识别出的文本
        scores = [line[1][1] for line in result]  # 置信度
        combined_txts = ''.join(txts)  # 将识别出的文本组合到一起
        st.markdown("### 识别结果：")
        # 显示图片结果
        st.markdown("#### 图片：")
        im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/aistudio/PaddleOCR/doc/fonts/simfang.ttf')  # 绘制检测框，显示文本和置信度
        im_show = Image.fromarray(im_show)
        st.image(im_show, channels="RGB")
        # im_show.save('result_self_1.jpg')  # 保存图片
        # 显示组合文本
        st.markdown("#### 文本：")
        st.write(combined_txts)
        # 将组合文本存入txt文件
        file_name = "test3.txt"  # 定义txt文件的名称
        # 使用with语句打开文件，以写入模式('w')打开
        # 'w'模式会覆盖文件内容，如果文件不存在则创建新文件
        # 如果想在文件已存在时追加内容，而不是覆盖，可以将模式从'w'改为'a'
        with open(file_name, 'w', encoding='utf-8') as file:
            # 将组合文本写入文件
            file.write(combined_txts)
        # 提示写入完成
        st.success(f"识别结果已成功写入到 {file_name}")

        # 输入访问令牌、域名地址、语言大模型种类
        st.text_input("请输入含有AI Studio的访问令牌（API_key）:",
                      key="API_key",
                      placeholder='48b27a1e9b01529229e08de0d57ba7d9f907a03d'
                      )
        st.text_input("请输入用星河大模型API服务的域名地址（base_URL）:",
                      key="base_URL",
                      placeholder="https://aistudio.baidu.com/llm/lmapi/v3"
                      )
        model = st.selectbox("请选择语言大模型：",
                             ('ernie-4.0-8k', 'ernie-4.0-turbo-8k', 'ernie-4.5-8k-preview', 'deepseek-r1')
                             )
        # 开始评阅按钮
        read_over = st.button("点击开始评阅")
        # 按钮按下
        if read_over:
            # 认证鉴权
            client = OpenAI(
                api_key=st.session_state.API_key,  # 含有AI Studio访问令牌的环境变量
                base_url=st.session_state.base_URL  # aistudio大模型api服务域名
            )
            # 输入准备
            question = "农业劳动有何意义？"
            answer = combined_txts
            score = 6
            system = "你是一个阅卷人"
            dialog = f"考试题目为：{question}，考生作答内容为：{answer}。该题满分为{score}分，应打多少分？"

            messages = [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': dialog}
            ]
            # 评阅内容生成
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                top_p=0.5,
                temperature=0.5,
                #stream=True,
            )
            # 一次性输出
            st.write(chat_completion.choices[0].message.content)
            # 流式输出
            #for chunk in chat_completion:
                #st.write(chunk.choices[0].delta.content or "", end="")

        else:
            st.stop()  # 退出

else:
    st.stop()  # 退出
