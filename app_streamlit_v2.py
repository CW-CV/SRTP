import streamlit as st
import numpy as np
import pandas as pd
import re
from PIL import Image
from paddleocr import PaddleOCR
from io import BytesIO
import docx
import pdfplumber
from openai import OpenAI
import matplotlib.pyplot as plt
import altair as alt
from collections import defaultdict
import traceback
import zipfile

from pylab import mpl

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["SimSun"]
mpl.rcParams["axes.unicode_minus"] = False


# ======================
# 初始化配置
# ======================
def init_settings():
    st.set_page_config(
        page_title="智能阅卷系统",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://education.example.com',
            'Report a bug': "mailto:support@example.com",
            'About': "智能阅卷系统 v2.0"
        }
    )


# ======================
# 核心工具函数
# ======================
@st.cache_resource(show_spinner="正在初始化OCR引擎...")
def load_ocr():
    return PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        det_model_dir='./models/det',
        rec_model_dir='./models/rec',
        #rec_model_dir='./infer',
        cls_model_dir='./models/cls'
    )


def validate_file(file, allowed_types):
    if not file:
        return False
    ext = file.name.split('.')[-1].lower()
    if ext not in allowed_types:
        raise ValueError(f"不支持的文件类型: {ext}")
    if file.size > 50 * 1024 * 1024:
        raise ValueError("文件大小超过50MB限制")
    return True


# ======================
# 文件处理模块
# ======================
class FileProcessor:
    @staticmethod
    def parse_uploaded_file(file):
        try:
            validate_file(file, ['pdf', 'docx', 'txt', 'jpg', 'png', 'jpeg'])
            file_type = file.name.split(".")[-1].lower()

            if file_type == "txt":
                return file.read().decode("utf-8"), None
            elif file_type == "docx":
                return FileProcessor._parse_docx(file), None
            elif file_type == "pdf":
                return FileProcessor._parse_pdf(file), None
            elif file_type in ['jpg', 'png', 'jpeg']:
                image_data = file.read()  # 保存原始图片数据
                text = FileProcessor._parse_image(BytesIO(image_data))
                return text, image_data
        except Exception as e:
            st.error(f"文件解析失败: {str(e)}")
            st.error(traceback.format_exc())
            return "", None

    @staticmethod
    def _parse_docx(file):
        try:
            header = file.getvalue()[:4]
            if header != b'PK\x03\x04':
                raise ValueError("无效的docx文件格式")

            doc = docx.Document(BytesIO(file.read()))
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        except zipfile.BadZipFile:
            st.error("文件损坏，请确认是有效的docx文件")
            return ""
        except Exception as e:
            st.error(f"DOCX解析失败：{str(e)}")
            return ""

    @staticmethod
    def _parse_pdf(file):
        text = []
        try:
            with pdfplumber.open(BytesIO(file.read())) as pdf:
                for page in pdf.pages:
                    text.append(page.extract_text(x_tolerance=1, y_tolerance=1))
            return "\n".join(filter(None, text))
        except Exception as e:
            st.error(f"PDF解析失败：{str(e)}")
            return ""

    @staticmethod
    def _parse_image(file):
        try:
            image = Image.open(file).convert('RGB')
            result = ocr.ocr(np.array(image), cls=True)
            if result and result[0]:
                return " ".join(line[1][0] for line in result[0] if line[1])
            return ""
        except Exception as e:
            st.error(f"图片解析失败: {str(e)}")
            st.error(traceback.format_exc())
            return ""


# ======================
# 参考答案解析模块
# ======================
class StandardAnswerParser:
    def __init__(self):
        self.category_pattern = re.compile(
            r'^(第[一二三四五六七八九十]+部分|[IVXLCDM]+、)[：:]?\s*(.+)'
        )
        self.question_pattern = re.compile(
            r'^(\d+[a-zA-Z]*)[\.、．]'
        )

    @st.cache_data(show_spinner="正在解析参考答案...")
    def parse(_self, files):
        categories = defaultdict(str)
        current_category = "未分类"
        content = []

        for file in files:
            text = FileProcessor.parse_uploaded_file(file)[0]
            for line in text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                if category_match := _self.category_pattern.match(line):
                    current_category = category_match.group(2)
                    content.append(f"\n【{current_category}】")
                    continue

                if question_match := _self.question_pattern.match(line):
                    q_num = question_match.group(1)
                    categories[q_num] = current_category

                content.append(line)

        return '\n'.join(content), dict(categories)


# ======================
# AI评分模块
# ======================
class GradingEngine:
    PROMPT_TEMPLATE = """
    【评分规则】
    1. 客观题：直接比对答案，错题扣分
    2. 主观题：按要点给分，每个要点分值相同
    3. 作文题：从结构（30%）、内容（40%）、语法（30%）三个维度评分
    4. 满分设置：{max_score}分

    【参考答案】
    {standard_answer}

    【学生答案】
    {student_answer}

    【输出要求】
    - 总分：{max_score}分制
    - 格式：
        总分：XX/{max_score}
        各题得分：
        1. [得分]/[满分] [简要分析]
        2. [得分]/[满分] [简要分析]
        ...
    - 改进建议：按题型分类给出
    """

    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def grade(self, standard, answer, max_score, model, top_p, temperature):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": self.PROMPT_TEMPLATE.format(
                        max_score=max_score,
                        standard_answer=standard,
                        student_answer=answer
                    )
                }],
                temperature=temperature,
                top_p=top_p,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"评分请求失败: {str(e)}")
            return ""


# ======================
# 界面组件
# ======================
class UIComponents:
    @staticmethod
    def render_sidebar():
        with st.sidebar:
            st.header("📤 材料上传区")
            UIComponents.file_manager()
            st.header("⚙️ 教师设置")
            UIComponents.settings_panel()
            st.header("🔑 API配置")
            st.text_input(
                "请输入含有AI Studio的访问令牌（API_key）:",
                key="API_key",
                placeholder='48b27a1e9b01529229e08de0d57ba7d9f907a03d'
            )
            st.text_input(
                "请输入用星河大模型API服务的域名地址（base_URL）:",
                key="base_URL",
                placeholder="https://aistudio.baidu.com/llm/lmapi/v3"
            )
            st.selectbox(
                "请选择语言大模型：",
                ('ernie-4.0-8k', 'ernie-4.0-turbo-8k', 'ernie-4.5-8k-preview', 'deepseek-r1'),
                key="model_choice"
            )
            st.slider(
                "top_p",
                0.0, 1.0, 0.8, 0.1,
                key="top_p"
            )
            st.slider(
                "temperature",
                0.0, 1.0, 0.3, 0.1,
                key="temperature"
            )

    @staticmethod
    def file_manager():
        with st.expander("📁 文件管理", expanded=True):
            st.warning("请确保上传的DOCX文件：\n1. 使用Office 2010+格式\n2.未设置密码保护\n3.大小不超过50MB")

            exam_files = st.file_uploader(
                "上传试题文件",
                type=["pdf", "docx", "jpg", "png"],
                accept_multiple_files=True,
                key="exam_uploader"
            )
            answer_files = st.file_uploader(
                "上传答题文件",
                type=["pdf", "docx", "jpg", "png"],
                accept_multiple_files=True,
                key="answer_uploader"
            )
            standard_files = st.file_uploader(
                "上传参考答案",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="standard_uploader"
            )

            # 自动保存上传文件到session_state
            for category, files in zip(
                    ['exams', 'answers', 'standards'],
                    [exam_files, answer_files, standard_files]
            ):
                if files:
                    existing_names = {f.name for f in st.session_state.uploaded_files[category]}
                    for f in files:
                        if f.name not in existing_names:
                            if category == 'standards' and st.session_state.uploaded_files['standards']:
                                if not st.confirm("重新上传标准答案将重置所有历史数据，确认继续？"):
                                    continue
                            st.session_state.uploaded_files[category].append(f)

            # 实时文件状态显示
            st.subheader("🗃 已上传文件")
            for category in ['exams', 'answers', 'standards']:
                if files := st.session_state.uploaded_files[category]:
                    st.write(f"📁 {category.capitalize()}:")
                    for i, file in enumerate(files[:5]):
                        cols = st.columns([6, 1])
                        cols[0].caption(f"{file.name[:20]}...")
                        if cols[1].button("×", key=f"del_{category}_{i}"):
                            if category == 'standards':
                                if st.confirm("删除标准答案文件将重置所有历史数据，确认继续？"):
                                    st.session_state.uploaded_files[category].remove(file)
                                    st.session_state.full_standard = ""
                                    st.session_state.question_categories = {}
                                    st.session_state.all_results = []
                                    st.rerun()
                            else:
                                st.session_state.uploaded_files[category].remove(file)
                                st.rerun()

    @staticmethod
    def settings_panel():
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.max_score = st.number_input(
                "试卷满分",
                min_value=0,
                max_value=1000,
                value=100,
                help="设置本试卷的总分"
            )
        with col2:
            st.session_state.subject = st.selectbox(
                "科目类型",
                options=["语文", "数学", "英语", "物理", "自定义"],
                index=0,
                help="选择学科以启用对应评分规则"
            )

    @staticmethod
    def display_compare_view():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("学生答案")
            st.text_area("学生答案", st.session_state.full_answer, height=400, key="student_answer_area")
        with col2:
            st.subheader("标准答案")
            st.text_area("标准答案", st.session_state.full_standard, height=400, key="standard_answer_area")

    @staticmethod
    def grading_section(grader):
        st.subheader("📝 智能评分")
        max_score = st.session_state.max_score

        if st.session_state.current_result:
            try:
                report_data = st.session_state.current_result["raw_output"]
                if isinstance(report_data, bytes):
                    report_data = report_data.decode("utf-8")
                st.download_button(
                    label="📥 下载评分报告",
                    data=report_data,
                    file_name="grading_report.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"下载评分报告失败: {str(e)}")

        if st.button("🚀 开始智能评分", type="primary"):
            if not st.session_state.get("API_key"):
                st.error("请先输入API Key")
                return
            if not st.session_state.get("base_URL"):
                st.error("请先输入Base URL")
                return

            with st.spinner("正在进行智能评分..."):
                response = grader.grade(
                    standard=st.session_state.full_standard,
                    answer=st.session_state.full_answer,
                    max_score=max_score,
                    model=st.session_state.get("model_choice", "ernie-4.0-8k"),
                    top_p=st.session_state.get("top_p", 0.8),
                    temperature=st.session_state.get("temperature", 0.3)
                )

            if response:
                total_score, question_scores = ScoreParser.parse(response)
                st.subheader(f"📊 {st.session_state.subject} 评分报告")

                try:
                    report_data = response.encode("utf-8")
                    st.download_button(
                        label="📥 下载原始评分报告",
                        data=report_data,
                        file_name="原始评分报告.txt",
                        mime="text/plain",
                        key="dl_raw_report"
                    )
                except Exception as e:
                    st.error(f"报告生成失败: {str(e)}")

                st.code(response, language="text")
                st.session_state.current_result = {
                    "raw_output": response,
                    "total_score": total_score,
                    "question_scores": question_scores
                }

        if st.session_state.current_result:
            result = st.session_state.current_result
            with st.expander("AI原始评分结果"):
                st.code(result["raw_output"], language="text")

            st.write("### 各题得分调整")
            cols = st.columns(6)
            modified_scores = {}
            for i, (q, s) in enumerate(result["question_scores"].items()):
                with cols[i % 6]:
                    modified_scores[q] = st.number_input(
                        f"{q}",
                        value=s,
                        min_value=0,
                        max_value=max_score,
                        step=1,
                        key=f"q_{q}",
                        on_change=UIComponents.update_total_score,
                        args=(modified_scores,)
                    )

            new_total = st.number_input(
                "修改总分",
                value=result['total_score'],
                min_value=0,
                max_value=max_score,
                step=1
            )

            if st.button("✅ 确认提交并保存", type="primary"):
                st.session_state.current_result["total_score"] = new_total
                st.session_state.current_result["question_scores"] = modified_scores
                st.session_state.all_results.append({
                    "total": new_total,
                    "subject": st.session_state.subject,
                    "max_score": st.session_state.max_score,
                    "details": modified_scores,
                    "original": result["raw_output"],
                    "question_categories": st.session_state.question_categories.copy()
                })
                st.session_state.uploaded_files['answers'] = []
                st.session_state.current_result = {}
                st.session_state.full_answer = ""
                st.success("评分已提交并保存！")
                st.rerun()

    @staticmethod
    def update_total_score(modified_scores):
        essay_categories = ['作文']
        total = sum(score for q, score in modified_scores.items()
                    if st.session_state.question_categories.get(q) not in essay_categories)
        st.session_state.current_result["total_score"] = total


# ======================
# 主程序
# ======================
def main():
    init_settings()
    global ocr
    ocr = load_ocr()
    parser = StandardAnswerParser()

    grader = GradingEngine(
        api_key=st.session_state.get("API_key", "5f078e2aaf490b9424b46c449975734a16c0799f"),
        base_url=st.session_state.get("base_URL", "https://aistudio.baidu.com/llm/lmapi/v3")
    )

    session_defaults = {
        'page': 'main',
        'all_results': [],
        'current_result': None,
        'uploaded_files': {'exams': [], 'answers': [], 'standards': []},
        'full_answer': "",
        'full_standard': "",
        'question_categories': {},
        'student_images': [],
        'exam_text': ""
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    with st.container():
        cols = st.columns([8, 1, 1])
        if st.session_state.page == 'main':
            cols[1].button("📈 学情分析", on_click=lambda: setattr(st.session_state, 'page', 'analysis'))
        elif st.session_state.page == 'analysis':
            cols[2].button("🏠 返回主页", on_click=lambda: setattr(st.session_state, 'page', 'main'))

    if st.session_state.page == 'main':
        render_main_page(parser, grader)
    elif st.session_state.page == 'analysis':
        render_analysis_page()


# ======================
# 页面渲染逻辑
# ======================
def render_main_page(parser, grader):
    UIComponents.render_sidebar()

    with st.container():
        st.title("智能阅卷系统")

        with st.expander("📥 答题卡处理", expanded=True):
            col1, col2 = st.columns([3, 2])
            with col1:
                if st.button("🔄 解析新答题卡",
                             help="上传新答题卡后点击此处进行解析",
                             disabled=not st.session_state.uploaded_files['answers']):
                    try:
                        st.session_state.full_answer = ""
                        st.session_state.student_images = []
                        st.session_state.exam_text = ""

                        exam_content = []
                        for f in st.session_state.uploaded_files['exams']:
                            try:
                                text, _ = FileProcessor.parse_uploaded_file(f)
                                exam_content.append(text)
                            except Exception as e:
                                st.error(f"试卷文件 {f.name} 解析失败: {str(e)}")
                        st.session_state.exam_text = "\n".join(exam_content)

                        student_text = []
                        student_images = []
                        for f in st.session_state.uploaded_files['answers']:
                            try:
                                text, image = FileProcessor.parse_uploaded_file(f)
                                if image:
                                    student_images.append(image)
                                if text:
                                    student_text.append(text)
                            except Exception as e:
                                st.error(f"答题文件 {f.name} 解析失败: {str(e)}")

                        st.session_state.full_answer = "\n".join(student_text)
                        st.session_state.student_images = student_images

                        st.success("答题卡解析成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"解析失败：{str(e)}")
                        st.error(traceback.format_exc())

            with col2:
                if st.session_state.full_answer or st.session_state.student_images:
                    st.metric("已解析内容",
                              f"{len(st.session_state.full_answer)}字符",
                              help="上次解析的答题卡内容长度")

        if st.session_state.get('student_images') or st.session_state.get('exam_text'):
            with st.expander("📄 答题卡与试卷对比", expanded=True):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("学生答题卡")
                    if st.session_state.student_images:
                        for i, img_bytes in enumerate(st.session_state.student_images):
                            try:
                                image = Image.open(BytesIO(img_bytes))
                                st.image(image, caption=f"答题卡第{i + 1}页", use_container_width=True)
                            except Exception as e:
                                st.error(f"图片显示失败: {str(e)}")
                    else:
                        st.text_area("文本答案", st.session_state.full_answer, height=400, key="student_answer_view")
                with col2:
                    st.subheader("试卷原文")
                    st.text_area("试卷内容", st.session_state.exam_text, height=400, key="exam_text_view")

        if not st.session_state.full_standard and st.session_state.uploaded_files['standards']:
            with st.spinner("正在自动解析标准答案..."):
                try:
                    st.session_state.full_standard, st.session_state.question_categories = parser.parse(
                        st.session_state.uploaded_files['standards']
                    )
                    st.success("标准答案初始化完成！")
                except Exception as e:
                    st.error(f"标准答案解析失败：{str(e)}")

        if st.session_state.full_standard:
            if st.session_state.full_answer or st.session_state.student_images:
                UIComponents.display_compare_view()
                UIComponents.grading_section(grader)
            else:
                st.info("请上传并解析新答题卡以开始评分")
        else:
            st.warning("请先上传标准答案文件")


def render_analysis_page():
    UIComponents.render_sidebar()

    if not st.session_state.all_results:
        st.warning("⚠️ 暂无评分数据，请先进行评分")
        return

    all_scores = [res["total"] for res in st.session_state.all_results]
    all_questions = defaultdict(list)
    for result in st.session_state.all_results:
        categories = result.get("question_categories", {})
        for q_num, score in result['details'].items():
            category = categories.get(q_num, '其他')
            all_questions[category].append(score)

    st.title(f"📊 {st.session_state.subject} 学情分析中心")

    max_total = max([res["max_score"] for res in st.session_state.all_results], default=100)

    st.subheader("📈 学情总览")
    col1, col2, col3 = st.columns(3)
    col1.metric("最高分", f"{max(all_scores)}/{max_total}")
    col2.metric("最低分", f"{min(all_scores)}/{max_total}")
    col3.metric("平均分", f"{np.mean(all_scores):.1f}/{max_total}")

    st.subheader("📊 分数分布分析")
    bin_step = 10 if max_total <= 100 else 20

    fig, ax = plt.subplots()
    ax.hist(all_scores,
            bins=np.arange(0, max_total + bin_step, bin_step),
            edgecolor='black')
    ax.set_xticks(np.arange(0, max_total + bin_step, bin_step))
    ax.set_xlabel(f"分数区间（满分{max_total}分）")
    ax.set_ylabel("人数")
    ax.set_title("分数分布直方图")
    st.pyplot(fig)

    st.subheader("📉 题型得分对比")
    df = pd.DataFrame({
        '题型': list(all_questions.keys()),
        '平均分': [np.mean(scores) for scores in all_questions.values()],
        '满分': max_total
    })
    df = df.sort_values(by='平均分', ascending=False)

    alt_chart = alt.Chart(df).mark_bar().encode(
        x='题型',
        y=alt.Y('平均分:Q', axis=alt.Axis(title='平均得分率%'), scale=alt.Scale(domain=[0, max_total])),
        tooltip=[
            alt.Tooltip('题型', title='题型'),
            alt.Tooltip('平均分:Q', title='平均分', format='.1f'),
            alt.Tooltip('满分:Q', title='题目满分')
        ]
    ).properties(width=600)

    st.altair_chart(alt_chart)

    fig, ax = plt.subplots()
    ax.pie(df['平均分'], labels=df['题型'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title("题型得分占比")
    st.pyplot(fig)


# ======================
# 其他优化组件
# ======================
class ScoreParser:
    @staticmethod
    def parse(content):
        total_pattern = re.compile(r"总分\s*[:：]\s*(\d+)\s*\/\s*(\d+)")
        question_pattern = re.compile(r"(\d+[a-zA-Z]*)\..*?(\d+)\s*\/\s*(\d+)")

        total_score = 0
        max_score = st.session_state.max_score
        question_scores = {}

        if total_match := total_pattern.search(content):
            total_score = int(total_match.group(1))
            max_score = int(total_match.group(2)) or max_score

        for match in question_pattern.finditer(content):
            q_num = match.group(1)
            score = int(match.group(2))
            question_scores[q_num] = score

        return total_score, question_scores


if __name__ == "__main__":
    main()
