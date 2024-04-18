#For Window OS
from os.path import join, dirname
from dotenv import load_dotenv
import time
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import (
    HumanMessage,
    SystemMessage
)
import os
import pandas as pd
import tiktoken
import time
import pytesseract
import pdfplumber
import tempfile
from pdf2image import convert_from_path

app = FastAPI()

tessdata_dir_config = r'./Tesseract-OCR/tessdata'
poppler_path = r'./poppler-0.68.0_x86/poppler-0.68.0/bin'
pytesseract.pytesseract.tesseract_cmd = r'./Tesseract-OCR/tesseract.exe'
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Configure OpenAI API
api_type = "azure"
api_version = os.environ.get("OPENAI_API_VERSION")
api_base = os.environ.get("OPENAI_API_BASE")
api_key = os.environ.get("Azure_API_KEY")
COST_PER_TOKEN = 0.0002

df = pd.read_excel("List_of_words_to_changes.xlsx",engine="openpyxl")
replacement_dict = dict(zip(df['ภาษาทั่วไป'].tolist(), df['ภาษาราชการ'].tolist()))

# Initialize a language model
llm = AzureChatOpenAI(deployment_name="test_content_creation_sarabun_usage",temperature=0.01, model="gpt-3.5-turbo-16k",
                       openai_api_key=api_key,openai_api_version=api_version,openai_api_base=api_base,openai_api_type=api_type)
refined_llm = AzureChatOpenAI(deployment_name="test_content_creation_sarabun_usage_small",temperature=0.01, model="gpt-3.5",
                       openai_api_key=api_key,openai_api_version=api_version,openai_api_base=api_base,openai_api_type=api_type)
gen_llm = AzureChatOpenAI(deployment_name="test_content_creation_sarabun_usage",temperature=0.1, model="gpt-3.5-turbo-16k",
                          openai_api_key=api_key,openai_api_version=api_version,openai_api_base=api_base,openai_api_type=api_type)

def split_text_to_chunks(text, max_chunk_size=2000):
    words = text.split(' ')
    chunks, current_chunk = [], ''

    for word in words:
        if len(current_chunk) + len(word) <= max_chunk_size:
            current_chunk += ' ' + word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def get_cost(tokens_used):
    return tokens_used * COST_PER_TOKEN

class ContentGenerationRequest(BaseModel):
    document_type: str
    prompts: str

app = FastAPI()

# origins = [
#     "http://localhost:4200",  # Angular runs on this port by default
#     # Add any other origins as needed
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

refine_template="""
การเขียนหนังสือราชการมีความสำคัญในการใช้ภาษาที่ชัดเจนและเป็นไปตามหลักการเพื่อให้ผู้อ่านเข้าใจได้ง่าย นี่คือสรุปและแยกประเด็นสำคัญในการเขียนหนังสือราชการ:
๑. การใช้ภาษาในการเขียนหนังสือราชการ:
  ๑.๑ การใช้สรรพนามแทนผู้มีหนังสือไป จึงนิยมใช้ชื่อส่วนราชการเป็นสรรพนามแทนผู้ลงนามในหนังสือไป จะไม่นิยมใช้ข้าพเจ้า หรือ กระผม เนื่องจากผู้ลงนามในหนังสือราชการ เป็นการ ลงนามในฐานะเป็นตัวแทนของส่วนราชการ ยกตัวอย่างเช่น
    'กรมสรรพากรพิจารณาแล้วเห็นว่า' 'กระทรวงแรงงานขอหารือว่า' กระทรวงมหาดไทยพิจารณาแล้วขอเรียนว่า ไม่นิยมใช้ 'ข้าพเจ้าขอเรียนว่า' 'กระผมขอเรียนว่า' เว้นแต่จะเป็นการลงนามในหนังสือในฐานะส่วนตัว
  ๑.๒ ใช้คำบุพบทให้ถูกต้องและไม่ยาวเกินไป
  การยกตัวอย่างโดยใช้คํา เช่น ได้แก่ อาทิในการยกตัวอย่างส่วนใหญ่ยังใช่กันสับสนที่ถูกต้องคือ
    “เช่น” ใช้ยกตัวอย่างคําต่าง ๆ ที่มีความหมายใกล้เคียงกัน แล้วลงท้ายด้วย ฯลฯ หรือเป็นต้น
    “ได้แก่” ไม่ใช้การยกตัวอย่าง จะต้องยกมาทั้งหมด
    “อาทิ” ยกมาเฉพาะที่สําคัญหรือลําดับต้น ๆ ไม่ต้องใช้ฯลฯ เพราะที่สําคัญมีเพียงเท่านั้น และไม่ควรใช้คําว่า
    “อาทิเช่น” เพราะ คําว่า อาทิและ เช่น มีความหมายเดียวกันคือการยกตัวอย่าง จึงไม่ควรใช้คําทั้งสองคํานี้ซ้อนกัน
  ๑.๓ ใช้คำเช่น "เช่น" เพื่อยกตัวอย่างและลงท้ายด้วย "ฯลฯ" หรือเป็นต้น
  ๑.๔ การขึ้นต้นด้วยกริยา จะชัดเจนดีเช่น ขออนุมัติ ขออนุญาตขอให้ แต่การขึ้นต้นด้วยคำนาม จะไม่ชัดเจน เช่น เครื่องพิมพ์ดีด
๒. การใช้เลขไทย:
  ๒.๑ ใช้เลขไทยทั้งฉบับในหนังสือราชการโดยให้ปรับเลขอารบิกเป็นเลขไทยเช่น 1,2,3,4,5,6,7,8,9 เป็น ๑,๒,๓,๔,๕,๖,๗,๗,๘ และ ๙ ตามลำดับ
  ๒.๒ เลขไทยในหนังสือราชการจะถูกใช้ให้ครบทุกที่ในเนื้อหาและไม่มีการใช้ตัวเลขรูปแบบอื่น นอกเหนือจากคำศัพท์
๓. ความนิยมที่ใช้ในวงราชการและความนิยมเฉพาะในหนังสือติดต่อในวงการราชการ:
  ๓.๑ ความนิยมในสรรพนาม: ในหนังสือติดต่อในนามส่วนราชการจะไม่ใช้คำสรรพนาม เช่น "ข้าพเจ้า" "ตน" "เขา" "เรา" "ท่าน" หรือ "ผม" แต่จะใช้ชื่อส่วนราชการแทน หรือไม่ก็ละเว้นไว้โดยไม่ระบุชื่อส่วนราชการ("")
  ๓.๒ ความนิยมในถ้อยคำสำนวน: มีการใช้ภาษาราชการในหนังสือราชการ การเชื่อมคำหรือประโยคด้วยคำบุพบทหรือคำสันธานที่มีความหมายเดียวกัน และไม่ใช้คำซ้ำกัน เช่น "ที่-ซึ่ง-อัน" ใช้แทนกันได้ทั้ง 3 คำ และ "และ-กับ-รวมทั้ง-ตลอดจน" ใช้แทนกันได้ทั้ง 4 คำนี้
  ๓.๓ ความนิยมในการเชื่อมคำประธานหรือกริยา: ใช้คำเชื่อม คำสุดท้ายคำเดียวเช่น "และ" หรือ "หรือ" ไม่ใส่คำเชื่อมทุกคำ เช่น ใช้ "ระบุคำสุดท้ายคำเดียว" และใช้กริยา "บัญญัติ" สำหรับพระราชบัญญัติ และใช้กริยา "กำหนด" สำหรับกฎระเบียบ
  ๓.๔ คำที่ใช้แทนกันได้และแทนกันไม่ได้: ใช้คำเช่น "กับ", "แก่", "แด่", "ต่อ", "และ", "หรือ", "และหรือ" เพื่อแทนกันได้หรือแทนกันไม่ได้
  ๓.๕ ความนิยมในคำเบา-คำหนักแน่น: ใช้คำเช่น "จะ" และ "จัก" โดย "จะ" ใช้ในกรณีทั่วไปและ "จัก" ใช้ในคำขู่ คำสั่ง หรือคำกำชับ และใช้คำเชื่องดับ "ควร", "พึง", "ย่อม", "ต้อง", และ "ให้" เพื่อแสดงความบังคับหรือเรียกร้อง
  ๓.๖ คำบังคับ-คำขอร้อง: ในการเขียนหนังสือที่มีถึงบุคคลหรือผู้ดำรงตำแหน่งที่ไม่ได้อยู่ในอำนาจเลขานุการ ใช้คำเช่น "ขอ", "เรียน", "ร้อง", "อุทธรณ์" เป็นต้น
  ๓.๗ คำที่เน้นความสำคัญ: ใช้คำเช่น "สำคัญ", "ยิ่ง", "สำคัญยิ่ง", "ด่วน", "ส่วนใหญ่", "สำคัญที่สุด", "แม่นยำ", "ไม่มีข้อผิดพลาด", "ไม่ถูกต้อง", "ไม่ได้ถูกต้อง" เพื่อเน้นความสำคัญของข้อความ
๔. ข้อควรระวังในการเขียนหนังสือราชการ:
  ๔.๑ หลีกเลี่ยงการใช้ภาษาที่ไม่เหมาะสม เช่น คำสแลง คำหยาบ หรือคำที่มีความหมายสะกดไม่ถูกต้อง
  ๔.๒ ใช้คำบรรยายในที่เหมาะสมเพื่อให้ผู้อ่านเข้าใจได้ง่าย และหลีกเลี่ยงคำพูดที่ซับซ้อนหรือมีความสับสน
  ๔.๓ ตรวจสอบคำศัพท์และคำต่าง ๆ เพื่อให้ถูกต้องทางไวยากรณ์และความหมายของคำศัพย์โดยยังอยู่ในกลุ่มคำราชการเท่านั้น
  ๔.๔ รักษาความเป็นส่วนตัวและความเป็นกลางในการเขียนหนังสือราชการ โดยไม่เปิดเผยข้อมูลส่วนตัวของบุคคลหรือเรื่องที่ไม่เกี่ยวข้อง
  ๔.๕ ควรปฏิบัติตามรูปแบบและแบบแผนการเขียนหนังสือราชการที่กำหนดโดยหน่วยงานที่เกี่ยวข้อง เช่น ระเบียบบัญญัติการเขียนหนังสือราชการในส่วนกำหนดของกระทรวงหรือหน่วยงานต่าง ๆ
  ๔.๖ เนื้อหาในการหนีังสือราชการห้ามใช้คำฟุ่มเฟือย คำซ้ำและคำกำกวนมากไป โดยให้เนื้อหาที่เขียนมาความกระชับและสั้นที่สุดแต่ยังมีความชัดเจนและลงรายละเอียดได้
๕. กระบวนการปรับคำ
  ๕.๑ พยายามปรับเนื้อหาให้มีความชัดเจนในเนื้อความ จุดประสงค์ และ มีวรรคตอนตามรูปแบบ
  ๕.๒ อ่านข้อตวามที่ผู้ใช้งานให้ลึกซึ้งและดูกลุ่มคำที่ละลำดับจนครบและค่อยเรียบเรียงให้ถูกต้องตามหลักภษาราชการ
  ๕.๓ พยายามปรับคำศัพท์ให้อยู่มนรูปแบบตามหลักภาษาในการเขียนหนังสือราชการโดยให้คงรูปหลักของข้อความเดิมไว้เหมือนเดิมและไม่ทำการตอบคำถามหรือเปลี่ยนแปลงรูปแบบประโยคใด ๆ แม้จะเป็นในรูปประโยคคำสั่งก็ตาม
"""

def refined_prompts_Gen(text):
    messages = [
        SystemMessage(content=f"คุณเป็น AI ที่มีความสามารถทางภาษาสามาถช่วยปรับชุดคำสั่งที่ผู้ใช้งานได้ถามมาให้สละสลวย ถูกต้อง และเข้าใจง่าย โดยเสริมเนื้อหาเพิ่มเติมและไม่ส่งผลกระทบต่อเนื้อหาที่ผู้ใช้งานต้องการ"),
        HumanMessage(content=f"ให้ปรับคำและเขียนชุดคำสั่งใหม่จากข้อความของผู้ใช้งานนี้ให้ดีขึ้นและชัดเจนขึ้น:\n'''{text}'''")
    ]
    refined_prompts = llm(messages).content
    return refined_prompts

def modify_words_with_dataframe(text):
    df = pd.read_excel("List_of_words_to_changes.xlsx",engine="openpyxl")
    words_to_modify = df['ภาษาทั่วไป'].tolist()
    replacement_words = df['ภาษาราชการ'].tolist()

    for word, replacement in zip(words_to_modify, replacement_words):
        index = text.find(word)
        while index != -1:
            text = text[:index] + replacement + text[index + len(word):]
            index = text.find(word, index + len(replacement))

    return text

def num_tokens_from_messages(message, model="gpt-3.5-turbo-0125"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(message, model="gpt-3.5-turbo-0125")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0125-Preview.")
        return num_tokens_from_messages(message, model="gpt-4-0125-Preview")
    elif model == "gpt-3.5-turbo-0125":
        tokens_per_message = 4
    elif model == "gpt-4-0125-Preview":
        tokens_per_message = 3
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    num_tokens += tokens_per_message
    num_tokens += len(encoding.encode(message))
    return num_tokens

def inner_refine_content(prompts):
    # Start time
    start_time = time.time()
    # Extract request data
    print("Start")

    # Modify the text using the dataframe
    modified_text = modify_words_with_dataframe(prompts)
    print("modify_raw_words")
    print(modified_text)

    refined_text = ""
    print("refined_text")
    # Define the PromptTemplate
    prompt_templates = [
        f"คุณเป็นผู้ช่วยเก่งภาษาไทยที่ช่วยปรับปรุงข้อความให้เป็นภาษาราชการ ตามหลักการใช้ภาษาราชการดังนี้:'''\n{refine_template}\n'''\n",
        f"เราได้เริ่มปรับข้อความตามหลักการใช้ภาษาราชการแล้ว, โปรดทำต่อจากนี้:\n"
    ]

    # Split the text into chunks that do not exceed the token limit
    chunks = split_text_to_chunks(modified_text, 8000 - num_tokens_from_messages(prompt_templates[0]) - 5) 

    for i, chunk in enumerate(chunks):
        messages = [
            SystemMessage(content=prompt_templates[min(i, 1)]),
            HumanMessage(content=f"ให้ปรับข้อความต่อไปนี้ให้เป็นไปตามภาษาราชการจากข้อความนี้ โดยไม่มีการเปลี่ยนแปลงเนื้อหาโดยให้คงรูปแบบของข้อความเดิมจากผู้ใช้งานไว้และไม่มีการเสริมเพิ่มเติมแต่อย่างใดเป็นอันขาด มิเช่นนั้นจะทำให้เกิดความสับสนต่อผู้ใช้งานได้:\n'''{chunk}'''")
        ]

        print("Processing chunk {}/{}".format(i+1, len(chunks)))
        response = llm(messages).content
        refined_text += response
    print(refined_text)
    print("sent_result")
    # End time
    end_time = time.time()

    # Calculate time taken
    time_taken = end_time - start_time

    # Calculate the tokens used
    tokens_used = num_tokens_from_messages(refined_text)
    print(tokens_used)

    # Calculate the cost
    cost = get_cost(tokens_used)
    return refined_text,tokens_used,cost,time_taken

@app.post("/refine-content")
async def refine_content(request: ContentGenerationRequest):
    # Start time
    start_time = time.time()
    print(request.prompts,request.document_type)

    # Extract request data
    prompts = request.prompts
    print("Start")

    # Modify the text using the dataframe
    modified_text = modify_words_with_dataframe(prompts)
    print("modify_raw_words")
    print(modified_text)

    refined_llm = AzureChatOpenAI(deployment_name="test_content_creation_sarabun_usage_small",temperature=0.01, model="gpt-3.5",
                       openai_api_key=api_key,openai_api_version=api_version,openai_api_base=api_base,openai_api_type=api_type)
    refined_text = ""

    prompt_templates = [
        f"คุณเป็นผู้ช่วยเก่งภาษาไทยที่ช่วยปรับปรุงข้อความให้เป็นภาษาราชการ ตามหลักการใช้ภาษาราชการดังนี้:'''\n{refine_template}\n'''\n กรุณาระบุคำหรือประโยคที่ต้องการให้ ChatGPT ปรับปรุงคำศัพท์เท่านั้น โดยไม่ต้องตอบคำถามหรือดำเนินการตามข้อความที่รับเข้ามา",
        f"เราได้เริ่มปรับข้อความตามหลักการใช้ภาษาราชการตามหลักการใช้ภาษาราชการดังนี้:'''\n{refine_template}\n'''\n แล้ว กรุณาระบุคำหรือประโยคที่ต้องการให้ ChatGPT ปรับปรุงคำศัพท์เท่านั้น โดยไม่ต้องตอบคำถามหรือดำเนินการตามข้อความที่รับเข้ามา, โปรดทำต่อจากนี้::\n"
    ]

    # Split the text into chunks that do not exceed the token limit
    chunks = split_text_to_chunks(modified_text, 1000)

    for i, chunk in enumerate(chunks):
        messages = [
            SystemMessage(content=prompt_templates[min(i, 1)]),
            HumanMessage(content=f"ให้ปรับข้อความต่อไปนี้ให้เป็นไปตามภาษาราชการจากข้อความนี้ โดยไม่มีการเปลี่ยนแปลงเนื้อหา คงรูปแบบข้อความเดิม และไม่มีการเสริมเพิ่มเติมแต่อย่างใด:\n'''{chunk}'''")
        ]

        print("Processing chunk {}/{}".format(i+1, len(chunks)))
        response = refined_llm(messages)
        refined_text += response.content

    # End time
    end_time = time.time()

    # Calculate time taken
    time_taken = end_time - start_time

    # Calculate the tokens used
    tokens_used = num_tokens_from_messages(refined_text)
    print(tokens_used)

    # Calculate the cost
    cost = get_cost(tokens_used)

    output = {
        'Contents': refined_text,
        'Time taken': str(time_taken) + " sec",
        'Tokens used': str(tokens_used) + " tokens",
        'Cost': str(cost) + " ฿"
    }
    print("sent result")
    return output

@app.post("/summarize-content")
async def summarize_content(request: ContentGenerationRequest):
    # Start time
    start_time = time.time()
    # Extract request data
    print(request.prompts,request.document_type)
    document_type = request.document_type
    prompts = request.prompts
    print("start")

    response = ""
    tokens_used = 0
    CHUNK_SIZE = 8192  # adjust this size according to your needs
    OVERLAP = 1024  # size of the overlap

    # Break down the text into smaller chunks with overlap
    chunks = [prompts[i:i+CHUNK_SIZE] for i in range(0, len(prompts), CHUNK_SIZE - OVERLAP)]
    print("Processing")

    prev_summary = ""  # to store previous summarization

    for chunk in chunks:
        combined_chunk = prev_summary + " " + chunk  # combine previous summary with new chunk
        messages = [
        SystemMessage(content=f"คุณเป็น AI ที่มีความสามารถทางภาษาสามาถช่วยปรับชุดคำสั่งที่ผู้ใช้งานได้ถามมาให้สละสลวย ถูกต้อง และเข้าใจง่าย โดยเสริมเนื้อหาเพิ่มเติมและไม่ส่งผลกระทบต่อเนื้อหาที่ผู้ใช้งานต้องการ"),
        HumanMessage(content=f"ช่วยสรุปเนื้อหาโดยเน้นประเด็นสำคัญดังต่อไปนี้และไม่ควรเกิน {document_type} คำจากข้อความต่อไปนี้: {combined_chunk}")
        ]

        # Response from Langchain OpenAI
        chunk_response = llm(messages).content
        prev_summary = chunk_response  # update the previous summary
        response = chunk_response
        # Calculate the tokens used
        chunk_tokens_used = num_tokens_from_messages(chunk_response)
        tokens_used += chunk_tokens_used

    print("sent_response")
    # End time
    end_time = time.time()

    # Calculate time taken
    time_taken = end_time - start_time

    # Calculate the cost
    cost = get_cost(tokens_used)
    output = {
        'Contents': response,
        'Time taken': str(time_taken) + " sec",
        'Tokens used': str(tokens_used) + " tokens",
        'Cost': str(cost) + " ฿"
    }
    print("sent result")
    return output

@app.post("/generate-content")
async def generate_content(request: ContentGenerationRequest):
    # Start time
    print("start")
    start_time = time.time()
    # Extract request data
    document_type = request.document_type
    prompts = request.prompts

    tokens_used = 0
    print("processing")
    # Acknowledgements_internal_messages_templates for each part
    Acknowledgements_internal_messages_templates = """เขียนเฉพาะส่วนของเนื้อหาแบบเรียงยาวไม่มีการ 'เรียน'ขึ้นต้นและคำลงท้ายใด ๆ หนังสือราชการภายในเป็นหนังสือติดต่อราชการภายในองค์กรหรือบุคคลากรในองค์กร ตามที่ผู้ใช้งานได้พิมพ์ไว้คร่าว ๆ แบ่งเป็น 3 ส่วนเท่านั้น:\
            ส่วนเหตุผล (เหตุที่มีหนังสือไป): กล่าวถึงเหตุผลหรือสาเหตุที่ต้องส่งหนังสือภายนอก พร้อมกับแหล่งที่มาของเอกสารที่ใส่มา โดยเน้นให้เห็นเหตุที่ต้องการตอบกลับ เช่น มีความประสงค์หรือมีการกระทำอะไรภายในองค์กร\
            ส่วนความประสงค์: แสดงความมุ่งหมายหรือวัตถุประสงค์ของการติดต่อ โดยต้องกล่าวถึงชื่อหน่วยงานภายในองค์กร แล้วนำเสนอเนื้อหาตามที่ผู้ใช้งานระบุไว้ โดยสามารถแบ่งเนื้อหาออกเป็นหัวข้อย่อยตามที่ผู้ใช้งานต้องการ\
            ส่วนสรุปความ: ในส่วนสุดท้ายของหนังสือ กรณีที่ผู้เขียนมีวัตถุประสงค์จะลื่อสารให้ ใช้คำว่า 'จึงเรียนมาเพื่อโปรดทราบ'เท่านั้น หรือมีเพิ่มเติมให้ขึ้นต้นด้วยกลุ่มคำดังเช่น 'จึงเรียนมาเพื่อโปรดพิจารณา' และอาจตามด้วย 'ดำเนินการต่อไปด้วย' สำหรับต้องการให้ทำอะไรต่อ หรือ 'อนุมัติด้วย' สำหรับต้องการให้เขียนลงเอกสารตกลงใด ๆ ที่แนบมา\
            และ'และแจ้งผลการพิจารณาให้ทราบด้วย' หรือ 'จึงเรียนมาเพื่อโปรดนำเสนอ...พิจารณาต่อไปด้วย' สำหรับกล่าวให้ทราบสั้น ๆ ขึ้นอยู่กับเป้าหมายของผู้เขียนต้องการจะสื่อสารผ่านหนังสือตอบรับราชการ แล้วเพิ่มคำอื่นที่เหมาะสม สรุปใจความโดยสั้น ๆ ที่คลุมเนื้อหาสำคัญของหนังสือ\
            และหลังจากสร้างเนื้อหาแล้ว ให้ปรับแต่งการเขียนหนังสือราชการในส่วนของเนื้อหาทั้ง 3 ส่วนที่ได้มาและไม่ต้องเขียนตามรูปแบบจดหมายใด ๆ ขอแค่ในส่วนของเนื้อหาของหนังสือราชการเท่านั้นโดยมีรูปแบบตาม\
            ชุดคำสั่งสำหรับการเขียนหนังสือภายนอกเพื่อสื่อสารระหว่างหน่วยงานหรือบุคคลโดยเขียนในรูปแบบที่ให้ดังนี้อย่างเคร่งครัดและให้แสดงผลลัพธ์ออกมาให้เหลือเพียง สามย่อหน้าเท่านั้น:
        '''
                (ตามหนังสือที่อ้างอิงถึง + (คำสั่งหรือเอกสารที่แนบมาและแหล่งที่มาของเอกสาร) + (เนื้อหาย่อของเรื่องนี้)) หรือ (ด้วย + (เนื้อหาย่อของเรื่องนี้)) (ส่วนเหตุผล)

                (ในการนี้ หรือ บัดนี้) + (ชื่อหน่วยงานหรือองค์กรตนเอง) + (เนื้อหาเกี่ยวกับเหตุที่ต้องการตอบกลับ) (เนื้อหาที่ผู้ใช้งานระบุไว้ในชุดคำสั่งเกี่ยวกับสื่อที่ต้องการและรูปแบบที่ผู้ใช้งานต้องการ) (ส่วนความประสงค์)

                (จึงเรียนมาเพื่อโปรดทราบ หรือ (จึงเรียนมาเพื่อโปรดพิจารณา)+(เนื้อหาที่ต้องการให้หน่วยงานตนเองและหน่วยงานที่จะส่งถึงต้องการพิจารราเรื่องในหนังสือ)) (ส่วนภาคสรุป)
        '''
        ตัวอย่างรูปแบบเขียนส่วนเนื้อหาเป็นดังนี้
        '''
                ตามที่กรมที่ดินได้แจ้งมาในหนังสือเลขที่ (ระบุหมายเลขหนังสือ) เรื่องการใช้พื้นที่สร้างถนนทางหลวง ตามแผนการก่อสร้างและแผนผังพื้นที่ที่แนบมาด้วย กรมโยธาธิการได้พิจารณาแล้วเห็นว่าเนื้อหาในหนังสือราชการของท่านไม่ตรงกับแผนการก่อสร้างที่จำกัดพื้นที่การสร้างไม่ให้ทำลายสิ่งแวดล้อมมากและเครื่องจักรการก่อสร้างได้รุกล้ำพื้นที่ป่าสงวน ดังนั้น กรมโยธาธิการขอให้ท่านส่งหนังสือตอบรับเพื่อให้กรมโยธาธิการพิจารณาโครงการก่อสร้างใหม่และจัดพื้นที่ให้เหมาะสม โดยมีเนื้อหาของหนังสือราชการดังนี้:

                (ชื่อหน่วยงานของท่าน) ขอเรียนว่า ตามที่กรมที่ดินได้แจ้งมาในหนังสือเลขที่ (ระบุหมายเลขหนังสือ) เรื่องการใช้พื้นที่สร้างถนนทางหลวง ตามแผนการก่อสร้างและแผนผังพื้นที่ที่แนบมาด้วย กรมโยธาธิการได้พิจารณาแล้วเห็นว่าเนื้อหาในหนังสือราชการของท่านไม่ตรงกับแผนการก่อสร้างที่จำกัดพื้นที่การสร้างไม่ให้ทำลายสิ่งแวดล้อมมากและเครื่องจักรการก่อสร้างได้รุกล้ำพื้นที่ป่าสงวน ดังนั้น กรมโยธาธิการขอให้ท่านส่งหนังสือตอบรับเพื่อให้กรมโยธาธิการพิจารณาโครงการก่อสร้างใหม่และจัดพื้นที่ให้เหมาะสม

                จึงเรียนมาเพื่อโปรดทราบ
        '''
        ทั้งนี้ขอให้พิจารณาการสร้างเนื้อหาออกมาได้อย่างถูกต้องก่อนแสดงผลลัพธ์ออกมาและไม่ใช้คำซ้ำมากเกินไป และคงรูปแบบการเขียนเนื้อหาให้อยู่ใรรูปวรรคตอนตามรูปแบบตัวอย่าง และเมื่อเขียนถึงส่วนสรุปความแล้วไม่ต้องพิมพ์ต่ออีก\
        โดยที่ต้องไม่ใช้สรรพนามว่า 'ข้าพเจ้า' อย่างเด็ดขาดโดยใช้ชื่อหน่วยงานหรือบุคคลที่ต้องการจะส่งให้แทนเท่านั้น
        """
    # Acknowledge_external_messages_templates for each part
    Acknowledge_external_messages_templates ="""เขียนเฉพาะส่วนของเนื้อหาแบบเรียงยาวไม่มีการ 'เรียน'ขึ้นต้นและคำลงท้ายใด ๆ สำหรับหนังสือราชการภายนอกเป็นหนังสือติดต่อราชการที่ใช้ในการติดต่อกับหน่วยงานราชการอื่นหรือบุคคลภายนอกตามที่ผู้ใช้งานได้พิมพ์ไว้คร่าว ๆ แบ่งเป็น 3 ส่วนเท่านั้น:\
            ส่วนเหตุผล (เหตุที่มีหนังสือไป): ในส่วนนี้จะกล่าวถึงเหตุผลหรือสาเหตุที่ต้องส่งหนังสือภายนอก เริ่มต้นด้วยคำว่า 'ตามที่' แล้วให้แจ้งชื่อหน่วยงานที่จะส่งถึง และอธิบายเหตุผลหรือเหตุที่เกี่ยวข้อง หากไม่มีหนังสืออ้างอิงให้ใช้คำว่า 'ด้วย' แล้วให้แจ้งชื่อหน่วยงานที่จะส่งถึง และอธิบายเหตุผลหรือเหตุที่เกี่ยวข้อง\
            ส่วนความประสงค์: ในส่วนนี้จะแสดงความมุ่งหมายหรือวัตถุประสงค์ของการติดต่อ ส่วนนี้เป็นส่วนสำคัญของหนังสือและควรขึ้นย่อหน้าใหม่ เริ่มต้นด้วยชื่อหน่วยงานที่ต้องการส่งถึง แล้วตามด้วยเนื้อหาที่ผู้ใช้งานระบุ โดยอาจมีหัวข้อย่อยหรือไม่ก็ได้ โดยจุดประสงค์ของการติดต่ออาจเป็นคำขอ คำสั่ง คำอนุมัติ หรือข้อตกลงก็ได้\
            ส่วนสรุปความ: ในส่วนสุดท้ายของหนังสือ กรณีที่ผู้เขียนมีวัตถุประสงค์จะลื่อสารให้ ใช้คำว่า 'จึงเรียนมาเพื่อโปรดทราบ'เท่านั้น หรือมีเพิ่มเติมให้ขึ้นต้นด้วยกลุ่มคำดังเช่น 'จึงเรียนมาเพื่อโปรดพิจารณา' และอาจตามด้วย 'ดำเนินการต่อไปด้วย' สำหรับต้องการให้ทำอะไรต่อ หรือ 'อนุมัติด้วย' สำหรับต้องการให้เขียนลงเอกสารตกลงใด ๆ ที่แนบมา\
            และ'และแจ้งผลการพิจารณาให้ทราบด้วย' หรือ 'จึงเรียนมาเพื่อโปรดนำเสนอ...พิจารณาต่อไปด้วย' สำหรับกล่าวให้ทราบสั้น ๆ ขึ้นอยู่กับเป้าหมายของผู้เขียนต้องการจะสื่อสารผ่านหนังสือตอบรับราชการ แล้วเพิ่มคำอื่นที่เหมาะสม สรุปใจความโดยสั้น ๆ ที่คลุมเนื้อหาสำคัญของหนังสือ\
            และหลังจากสร้างเนื้อหาแล้ว ให้ปรับแต่งการเขียนหนังสือราชการในส่วนของเนื้อหาทั้ง 3 ส่วนที่ได้มาและไม่ต้องเขียนตามรูปแบบจดหมายใด ๆ ขอแค่ในส่วนของเนื้อหาของหนังสือราชการเท่านั้นโดยมีรูปแบบตาม\
            ชุดคำสั่งสำหรับการเขียนหนังสือภายนอกเพื่อสื่อสารระหว่างหน่วยงานหรือบุคคลโดยเขียนในรูปแบบที่ให้ดังนี้อย่างเคร่งครัดและให้แสดงผลลัพธ์ออกมาให้เหลือเพียง สามย่อหน้าเท่านั้น:
        '''
                (ตามที่ (ชื่อหน่วยงานที่จะส่งถึง) + (คำสั่งหรือเอกสารที่แนบมาและแหล่งที่มาของเอกสาร) + (เนื้อหาย่อของเรื่องนี้)) หรือ (ด้วย(ชื่อหน่วยงานที่จะส่งถึง)(เนื้อหาย่อของเรื่องนี้)) (ส่วนเหตุผล)

                (ชื่อหน่วยงานหรือองค์กรตนเอง) + (เนื้อหาเกี่ยวกับเหตุที่ต้องการตอบกลับ) + (ชื่อหน่วยงานที่ต้องการส่งถึง) + (เนื้อหาที่ผู้ใช้งานระบุไว้ในชุดคำสั่งเกี่ยวกับสื่อที่ต้องการและรูปแบบที่ผู้ใช้งานต้องการ) (ส่วนความประสงค์)

                (จึงเรียนมาเพื่อโปรดทราบ หรือ (จึงเรียนมาเพื่อโปรดพิจารณา) + (เนื้อหาที่ต้องการให้หน่วยงานตนเองและหน่วยงานที่จะส่งถึงต้องการพิจารราเรื่องในหนังสือ)) (ส่วนภาคสรุป)
        '''
        ตัวอย่างรูปแบบเขียนส่วนเนื้อหาเป็นดังนี้
        '''
                ตามที่ (ชื่อหน่วยงานที่จะส่งถึง) ได้ส่งหนังสือของท่านมาเพื่อตอบรับเรื่องการปรับพื้นที่การเกษตร โดยมีเอกสารแนบเป็นแผนผังพื้นที่การเกษตรล่าสุด ที่เป็นการปรับพื้นที่การเกษตรใหม่ที่ไปทับซ้อนกับพื้นที่ป่าสงวน และเริ่มมีผู้คนบุกรุกเข้ามาในพื้นที่ป่าสงวนโดยไม่ได้รับอนุญาต ทำให้สิ่งแวดล้อมและสัตว์ป่าสงวนได้รับผลกระทบและลดจำนวนลงจำนวนมาก และบางรายได้เรียกร้องมาที่กรมป่าไม้เนื่องจากมีสัตว์ป่าได้รุกล้ำเข้าพื้นที่เกษตรและทำลสยผลผลิตการเกษตร

                ด้วยเหตุนี้ กรมการเกษตรจึงขอเชิญกรมป่าไม้และกรมที่ดินมาพูดคุยเพื่อวางผังพื้นที่การเกษตรให้เหมาะสมและไม่กระทบต่อสิ่งแวดล้อมและสัตว์ป่าสงวนในพื้นที่ โดยให้คำแนะนำและแนวทางการปรับปรุงพื้นที่การเกษตรให้เหมาะสมและไม่กระทบต่อสิ่งแวดล้อมและสัตว์ป่าสงวนในพื้นที่

                จึงเรียนมาเพื่อโปรดพิจารณาดำเนินการต่อไปด้วย
        '''
        ทั้งนี้ขอให้พิจารณาการสร้างเนื้อหาออกมาได้อย่างถูกต้องก่อนแสดงผลลัพธ์ออกมาและไม่ใช้คำซ้ำมากเกินไป และคงรูปแบบการเขียนเนื้อหาให้อยู่ใรรูปวรรคตอนตามรูปแบบตัวอย่าง และเมื่อเขียนถึงส่วนสรุปความแล้วไม่ต้องพิมพ์ต่ออีก\
        โดยที่ต้องไม่ใช้สรรพนามว่า 'ข้าพเจ้า' อย่างเด็ดขาดโดยใช้ชื่อหน่วยงานหรือบุคคลที่ต้องการจะส่งให้แทนเท่านั้น
        """
    # others_template for each part
    others_template = """เขียนเฉพาะส่วนของเนื้อหาแบบเรียงยาวไม่มีการ 'เรียน'ขึ้นต้นและคำลงท้ายใด ๆ หนังสือราชการภายนอกเป็นหนังสือติดต่อราชการที่ใช้ในการติดต่อกับหน่วยงานราชการอื่นหรือบุคคลภายนอกตามที่ผู้ใช้งานได้พิมพ์ไว้คร่าว ๆ แบ่งเป็น 3 ส่วนเท่านั้น:\
            ส่วนเหตุผล (เหตุที่มีหนังสือไป): ในส่วนนี้จะกล่าวถึงเหตุผลหรือสาเหตุที่ต้องส่งหนังสือภายนอก เริ่มต้นด้วยคำว่า 'ตามที่' แล้วให้แจ้งชื่อหน่วยงานที่จะส่งถึง และอธิบายเหตุผลหรือเหตุที่เกี่ยวข้อง หากไม่มีหนังสืออ้างอิงให้ใช้คำว่า 'ด้วย' แล้วให้แจ้งชื่อหน่วยงานที่จะส่งถึง และอธิบายเหตุผลหรือเหตุที่เกี่ยวข้องฃ\
            ส่วนความประสงค์: ในส่วนนี้จะแสดงความมุ่งหมายหรือวัตถุประสงค์ของการติดต่อ ส่วนนี้เป็นส่วนสำคัญของหนังสือและควรขึ้นย่อหน้าใหม่ เริ่มต้นด้วยชื่อหน่วยงานที่ต้องการส่งถึง แล้วตามด้วยเนื้อหาที่ผู้ใช้งานระบุ โดยอาจมีหัวข้อย่อยหรือไม่ก็ได้ โดยจุดประสงค์ของการติดต่ออาจเป็นคำขอ คำสั่ง คำอนุมัติ หรือข้อตกลงก็ได้\
            ส่วนสรุปความ: ในส่วนสุดท้ายของหนังสือ กรณีที่ผู้เขียนมีวัตถุประสงค์จะลื่อสารให้ ใช้คำว่า 'จึงเรียนมาเพื่อโปรดทราบ'เท่านั้น หรือมีเพิ่มเติมให้ขึ้นต้นด้วยกลุ่มคำดังเช่น 'จึงเรียนมาเพื่อโปรดพิจารณา' และอาจตามด้วย 'ดำเนินการต่อไปด้วย' สำหรับต้องการให้ทำอะไรต่อ หรือ 'อนุมัติด้วย' สำหรับต้องการให้เขียนลงเอกสารตกลงใด ๆ ที่แนบมา\
            และ'และแจ้งผลการพิจารณาให้ทราบด้วย' หรือ 'จึงเรียนมาเพื่อโปรดนำเสนอ...พิจารณาต่อไปด้วย' สำหรับกล่าวให้ทราบสั้น ๆ ขึ้นอยู่กับเป้าหมายของผู้เขียนต้องการจะสื่อสารผ่านหนังสือตอบรับราชการ แล้วเพิ่มคำอื่นที่เหมาะสม สรุปใจความโดยสั้น ๆ ที่คลุมเนื้อหาสำคัญของหนังสือ\
            และหลังจากสร้างเนื้อหาแล้ว ให้ปรับแต่งการเขียนหนังสือราชการในส่วนของเนื้อหาทั้ง 3 ส่วนที่ได้มาและไม่ต้องเขียนตามรูปแบบจดหมายใด ๆ ขอแค่ในส่วนของเนื้อหาของหนังสือราชการเท่านั้นโดยมีรูปแบบตาม\
            ชุดคำสั่งสำหรับการเขียนหนังสือภายนอกเพื่อสื่อสารระหว่างหน่วยงานหรือบุคคลโดยเขียนในรูปแบบที่ให้ดังนี้อย่างเคร่งครัดและให้แสดงผลลัพธ์ออกมาให้เหลือเพียง สามย่อหน้าเท่านั้น:
        '''
            (ตามที่ (ชื่อหน่วยงานที่จะส่งถึง) + (คำสั่งหรือเอกสารที่แนบมาและแหล่งที่มาของเอกสาร) + (เนื้อหาย่อของเรื่องนี้)) หรือ (ด้วย(ชื่อหน่วยงานที่จะส่งถึง)(เนื้อหาย่อของเรื่องนี้)) (ส่วนเหตุผล)

            (ชื่อหน่วยงานหรือองค์กรตนเอง) + (เนื้อหาเกี่ยวกับเหตุที่ต้องการตอบกลับ) + (ชื่อหน่วยงานที่ต้องการส่งถึง) + (เนื้อหาที่ผู้ใช้งานระบุไว้ในชุดคำสั่งเกี่ยวกับสื่อที่ต้องการและรูปแบบที่ผู้ใช้งานต้องการ) (ส่วนความประสงค์)

            (จึงเรียนมาเพื่อโปรดทราบ หรือ (จึงเรียนมาเพื่อโปรดพิจารณา) + (เนื้อหาที่ต้องการให้หน่วยงานตนเองและหน่วยงานที่จะส่งถึงต้องการพิจารราเรื่องในหนังสือ)) (ส่วนภาคสรุป)
        '''
        ตัวอย่างรูปแบบเขียนส่วนเนื้อหาเป็นดังนี้
        '''
                ตามที่ (ชื่อหน่วยงานที่จะส่งถึง) ได้ส่งหนังสือของท่านมาเพื่อตอบรับเรื่องการปรับพื้นที่การเกษตร โดยมีเอกสารแนบเป็นแผนผังพื้นที่การเกษตรล่าสุด ที่เป็นการปรับพื้นที่การเกษตรใหม่ที่ไปทับซ้อนกับพื้นที่ป่าสงวน และเริ่มมีผู้คนบุกรุกเข้ามาในพื้นที่ป่าสงวนโดยไม่ได้รับอนุญาต ทำให้สิ่งแวดล้อมและสัตว์ป่าสงวนได้รับผลกระทบและลดจำนวนลงจำนวนมาก และบางรายได้เรียกร้องมาที่กรมป่าไม้เนื่องจากมีสัตว์ป่าได้รุกล้ำเข้าพื้นที่เกษตรและทำลสยผลผลิตการเกษตร

                ด้วยเหตุนี้ กรมการเกษตรจึงขอเชิญกรมป่าไม้และกรมที่ดินมาพูดคุยเพื่อวางผังพื้นที่การเกษตรให้เหมาะสมและไม่กระทบต่อสิ่งแวดล้อมและสัตว์ป่าสงวนในพื้นที่ โดยให้คำแนะนำและแนวทางการปรับปรุงพื้นที่การเกษตรให้เหมาะสมและไม่กระทบต่อสิ่งแวดล้อมและสัตว์ป่าสงวนในพื้นที่

                จึงเรียนมาเพื่อโปรดพิจารณาดำเนินการต่อไปด้วย
        '''
        ทั้งนี้ขอให้พิจารณาการสร้างเนื้อหาออกมาได้อย่างถูกต้องก่อนแสดงผลลัพธ์ออกมาและไม่ใช้คำซ้ำมากเกินไป และคงรูปแบบการเขียนเนื้อหาให้อยู่ใรรูปวรรคตอนตามรูปแบบตัวอย่าง และเมื่อเขียนถึงส่วนสรุปความแล้วไม่ต้องพิมพ์ต่ออีก\
        โดยที่ต้องไม่ใช้สรรพนามว่า 'ข้าพเจ้า' อย่างเด็ดขาดโดยใช้ชื่อหน่วยงานหรือบุคคลที่ต้องการจะส่งให้แทนเท่านั้น
        """

    match document_type:
        case "หนังสือตอบรับภายใน":
            part_templates = Acknowledgements_internal_messages_templates
        case "หนังสือตอบรับภายนอก":
            part_templates = Acknowledge_external_messages_templates
        case "หนังสือราชการอื่น ๆ":
            part_templates = others_template

    print("generate")

    # Prepare the message
    messages = [
        SystemMessage(content=f"คุณเป็น AI ที่มีความสามารถทางภาษาสามาถช่วยปรับชุดคำสั่งที่ผู้ใช้งานได้ถามมาให้สละสลวย ถูกต้อง เข้าใจง่าย และไม่เกินสามย่อหน้า โดยเสริมเนื้อหาเพิ่มเติมและไม่ส่งผลกระทบต่อเนื้อหาที่ผู้ใช้งานต้องการ มีกระบวนการคิดและหลักการเขียนเนื้อหาสำหรับหนังสือตอบรับราชการดังนี้\
                      {part_templates}\n เมื่อเขียนเสร็จแล้วกรุณาช่วยตรวจสอบการใช้ภาษาสำหรับการเขียนหนังสือราชการตามหลักการต่อไปนี้: {refine_template}"),
        HumanMessage(content=f"ช่วยสร้างเนื้อหาที่ผู้ใช้งานต้องจะให้เขียนในหนังสือราชการและปรับเนื้อหาให้มีรูปแบบตามระบบได้กำหนดจากชุดคำสั่งของผู้ใช้งานดังนี้: {prompts} ให้สร้างเนื้อหาออกมาเพียง สามย่อหน้าเท่านั้น")
    ]
    # Generate the part
    tokens_used += num_tokens_from_messages(f"คุณเป็น AI ที่มีความสามารถทางภาษาสามาถช่วยปรับชุดคำสั่งที่ผู้ใช้งานได้ถามมาให้สละสลวย ถูกต้อง เข้าใจง่าย และไม่เกินสามย่อหน้า โดยเสริมเนื้อหาเพิ่มเติมและไม่ส่งผลกระทบต่อเนื้อหาที่ผู้ใช้งานต้องการ มีกระบวนการคิดและหลักการเขียนเนื้อหาสำหรับหนังสือตอบรับราชการดังนี้\
                      {part_templates}\n เมื่อเขียนเสร็จแล้วกรุณาช่วยตรวจสอบการใช้ภาษาสำหรับการเขียนหนังสือราชการตามหลักการต่อไปนี้: {refine_template}")
    tokens_used += num_tokens_from_messages(f"ช่วยสร้างเนื้อหาที่ผู้ใช้งานต้องจะให้เขียนในหนังสือราชการและปรับเนื้อหาให้มีรูปแบบตามระบบได้กำหนดจากชุดคำสั่งของผู้ใช้งานดังนี้: {prompts} ให้สร้างเนื้อหาออกมาเพียง สามย่อหน้าเท่านั้น")
    response = gen_llm(messages).content
    print(response)
    # Calculate the tokens used
    tokens_used += num_tokens_from_messages(response)
    # End time
    end_time = time.time()
    print("sent_result")
    # Calculate time taken
    time_taken = end_time - start_time
    # Calculate the cost
    cost = get_cost(tokens_used)
    # Prepare the output in JSON format
    output = {
        'Contents': response,
        'Time taken': str(time_taken)+" sec",
        'Tokens used': str(tokens_used)+" tokens",
        'Cost': str(cost)+" ฿"
    }
    print("sent result")
    return output

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # read file from angular and then exprort in text to angular to put in text input box.
    with tempfile.TemporaryDirectory() as tempdir:
        pdf_path = os.path.join(tempdir, 'temp.pdf')
        with open(pdf_path, 'wb') as out_file:
            out_file.write(await file.read())

        with pdfplumber.open(pdf_path) as pdf:
            text = ' '.join(page.extract_text() for page in pdf.pages)
        if text.strip() == "":
            images = convert_from_path(pdf_path)
            ocr_text = ''
            for i in range(len(images)):
                page_content = pytesseract.image_to_string(images[i],lang="tha")
                page_content = '\n'.format(i+1) + page_content
                ocr_text = ocr_text + ' ' + page_content
            text = ocr_text
        else:
            text = text

    return {'text': text}

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8150)

