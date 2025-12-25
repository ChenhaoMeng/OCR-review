import streamlit as st
import pandas as pd
import json
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
from openai import OpenAI

# --- 全局配置 ---
# 1. 尝试从 Streamlit Secrets 读取 Key (生产环境)
# 2. 如果没有，则留空 (会报错提示)
try:
    API_KEY = st.secrets["SJTU_API_KEY"]
except Exception:
    API_KEY = "" # 本地测试时，如果没配置 .streamlit/secrets.toml 会走这里

API_BASE = "https://models.sjtu.edu.cn/api/v1"

st.set_page_config(page_title="德语教材 AI 解析器", layout="wide")

# ... (其余代码保持不变) ...

# 在调用 API 之前增加一个检查
if not API_KEY:
    st.error("未检测到 API Key。请在 Streamlit Cloud Secrets 中配置 'SJTU_API_KEY'。")
    st.stop()

# ... (确保 extract_text_with_vision 和 analyze_grammar 函数里使用的是这个 API_KEY) ...


# --- 1. 辅助函数：PDF 页转 Base64 图片 ---
def pdf_page_to_base64(uploaded_file, page_number=0):
    """
    将上传的 PDF 文件的指定页面转换为 Base64 编码的图像字符串。
    这样可以直接发送给 API。
    """
    # 使用 PyMuPDF 打开文件流
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    if page_number >= len(doc):
        return None, "页码超出范围"
        
    page = doc.load_page(page_number)
    
    # 将页面渲染为像素图 (dpi=150 保证清晰度且不会导致图片过大消耗太多 Token)
    pix = page.get_pixmap(dpi=150)
    
    # 转换为 PIL Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # 转换为 Base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str, img

# --- 2. AI 视觉提取文本 (OCR) ---
def extract_text_with_vision(base64_image):
    """
    使用 Qwen3-VL-32B 模型从图片中提取德语文本。
    """
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    try:
        response = client.chat.completions.create(
            model="Qwen3-VL-32B",  # 使用视觉模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "请将这张图片中的德语文本完整地转录出来。只输出德语内容，不要包含其他解释。如果包含标题和正文，请按顺序转录。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000  # 限制输出长度
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# --- 3. AI 语法分析 (纯文本处理) ---
@st.cache_data(show_spinner=False)
def analyze_grammar(text):
    """
    使用 DeepSeek-V3 分析提取出来的文本
    """
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)
    
    # 截断文本以防止超过 Token 限制 (每周/每分钟限制)
    # 假设每分钟只能跑 3000 token，我们这里尽量保守
    safe_text = text[:800] 
    
    prompt = f"""
    分析以下德语文本。提取重点单词（过滤简单介词和冠词）。
    请严格按照 JSON 格式返回列表，不要使用 Markdown 格式。
    字段: word(原词), pos(词性), meaning(中文), usage(语法/搭配), example(极短例句)。
    
    文本: "{safe_text}"
    """

    try:
        response = client.chat.completions.create(
            model="DeepSeek-V3-685B", # 文本分析能力更强
            messages=[
                {"role": "system", "content": "你是一个输出纯 JSON 的德语助教。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        clean_content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_content)
    except Exception as e:
        return {"error": str(e)}

# --- 4. 界面逻辑 ---
st.title("🇩🇪 德语教材 OCR 智能助手")
st.caption("支持 PDF 扫描件：利用 Qwen3-VL 识图 -> DeepSeek-V3 分析")

# 侧边栏
with st.sidebar:
    st.header("文件上传")
    uploaded_file = st.file_uploader("上传教材 PDF", type=["pdf"])
    page_num = st.number_input("选择页码 (从0开始)", min_value=0, value=0, step=1)
    
    st.divider()
    st.warning("⚠️ 资源限制提示：\n每分钟限制 3000 Tokens。\n建议每次只分析一页，操作间隔 30 秒以上。")

if uploaded_file is not None:
    # 1. 转换图片
    with st.spinner("正在渲染 PDF 页面..."):
        # 必须重置文件指针，否则切换页码时会报错
        uploaded_file.seek(0) 
        base64_img, pil_img = pdf_page_to_base64(uploaded_file, page_num)
    
    if pil_img:
        # 展示图片
        st.image(pil_img, caption=f"第 {page_num} 页预览", use_container_width=True)
        
        # 按钮触发 OCR 和分析
        if st.button("🔍 提取文字并分析语法", type="primary"):
            
            # 2. 视觉提取 (耗费 Token)
            with st.spinner("正在使用 Qwen3-VL 读取图片文字..."):
                extracted_text = extract_text_with_vision(base64_img)
            
            if "Error" in extracted_text:
                st.error("图片识别失败，请重试。")
                st.error(extracted_text)
            else:
                st.subheader("📄 提取的文本")
                st.text_area("OCR 结果 (可手动修正)", value=extracted_text, height=150, key="ocr_text")
                
                # 3. 语法分析 (耗费 Token)
                # 使用 session_state 中的值，允许用户修正 OCR 错误后再分析
                text_to_analyze = st.session_state.ocr_text if "ocr_text" in st.session_state else extracted_text
                
                with st.spinner("正在使用 DeepSeek-V3 解析语法..."):
                    analysis_result = analyze_grammar(text_to_analyze)
                
                if "error" in analysis_result:
                    st.error(f"分析出错: {analysis_result['error']}")
                else:
                    st.subheader("🎓 语法详解")
                    df = pd.DataFrame(analysis_result)
                    st.dataframe(
                        df, 
                        column_config={
                            "word": "单词", "pos": "词性", "meaning": "中文含义", 
                            "usage": "用法/搭配", "example": "例句"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
    else:
        st.error("无法加载该页面，可能页码超出了文件范围。")
else:
    st.info("请先在左侧上传 PDF 文件。")
