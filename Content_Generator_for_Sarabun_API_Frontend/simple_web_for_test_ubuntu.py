# main.py
import streamlit as st
import requests
import docx2txt
import io
import tempfile
import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

API_ENDPOINT_GEN = os.getenv("API_ENDPOINT_GEN", "http://localhost:8150/generate-content")
API_ENDPOINT_REFINE = os.getenv("API_ENDPOINT_REFINE", "http://localhost:8150/refine-content")
API_ENDPOINT_SUM = os.getenv("API_ENDPOINT_SUM", "http://localhost:8150/summarize-content")

class ContentGenerationRequest:
    def __init__(self, document_type, prompts):
        self.document_type = document_type
        self.prompts = prompts

    def generate_content(self):
        response = requests.post(API_ENDPOINT_GEN, json={
            'document_type': self.document_type,
            'prompts': self.prompts,
        })
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None

    def refine_content(self):
        response = requests.post(API_ENDPOINT_REFINE, json={
            'document_type': self.document_type,
            'prompts': self.prompts,
        })
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    def summarize_content(self):
        response = requests.post(API_ENDPOINT_SUM, json={
            'document_type': self.document_type,
            'prompts': self.prompts,
        })
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None

def convert_doc_to_text(uploaded_file):
    # Save uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()

    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(temp_file.name) as pdf:
            text = " ".join([page.extract_text() for page in pdf.pages])
        if text.strip() == "":
            images = convert_from_path(temp_file.name)
            ocr_text = ''
            for i in range(len(images)):
                page_content = pytesseract.image_to_string(images[i],lang="tha")
                page_content = '\n'.format(i+1) + page_content
                ocr_text = ocr_text + ' ' + page_content
            text = ocr_text
        else:
            text = text
    elif uploaded_file.name.endswith('.docx'):
        text = docx2txt.process(io.BytesIO(uploaded_file.read()))
    else:
        st.error("Invalid file format: File must be .pdf or .docx")
        return None

    # Remove temporary file
    os.unlink(temp_file.name)

    return text

def main():
    st.title("AI Document Processing")

    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Content Generator", "Document Refinement","Summarized Documents"])
    if app_mode == "Content Generator":
        st.subheader("Input setting")
        document_type = st.selectbox("Document Type", options=['หนังสือตอบรับภายนอก', 'หนังสือตอบรับภายใน','หนังสือราชการอื่น ๆ'])
        prompt = st.text_area("Enter your prompt:")
        if st.button("Generate Content"):
            request_data = ContentGenerationRequest(
                document_type=document_type,
                prompts=prompt,
            )
            result = request_data.generate_content()
            if result is not None:
                st.subheader("Output")
                st.json(result)
                st.text(result["Contents"])

    elif app_mode == "Document Refinement":
        st.subheader("Input setting")
        refine_option = st.selectbox("Refinement Option", options=["Upload File", "Text Input"])
        if refine_option == "Upload File":
            uploaded_file = st.file_uploader("Upload .pdf or .docx document")

            if uploaded_file is not None:
                document_text = convert_doc_to_text(uploaded_file)
                if document_text is not None:
                    if st.button("Refine Document"):
                        request_data = ContentGenerationRequest(
                            document_type="not_use",
                            prompts=document_text,
                        )
                        result = request_data.refine_content()
                        if result is not None:
                            st.subheader("Output")
                            st.json(result)
                            st.text(result["Contents"])

        elif refine_option == "Text Input":
            st.subheader("Input Parameters")
            prompts = st.text_area("Enter your document texts:")
            if st.button("Refine Document"):
                request_data = ContentGenerationRequest(
                    document_type="not_use",
                    prompts=prompts,
                )
                result = request_data.refine_content()
                if result is not None:
                    st.subheader("Output")
                    st.json(result)
                    st.text(result["Contents"])

    elif app_mode == "Summarized Documents":
        refine_option = st.selectbox("Refinement Option", options=["Upload File", "Text Input"])
        Summarized_words = st.number_input("Enter number of words that should summarized out",min_value=200,max_value=800)
        if refine_option == "Upload File":
            uploaded_file = st.file_uploader("Upload .pdf or .docx document")
            if uploaded_file is not None:
                document_text = convert_doc_to_text(uploaded_file)
                st.text(document_text)
                if document_text is not None:
                    if st.button("Summarize Document"):
                        request_data = ContentGenerationRequest(
                            document_type="not_use",
                            prompts=document_text,
                        )
                        result = request_data.summarize_content()
                        if result is not None:
                            st.subheader("Output")
                            st.json(result)
                            st.text(result["Contents"])

        elif refine_option == "Text Input":
            st.subheader("Input Parameters")
            prompts = st.text_area("Enter your texts")
            if st.button("Summarize Document"):
                request_data = ContentGenerationRequest(
                document_type=str(Summarized_words),
                    prompts=prompts,
                )
                result = request_data.summarize_content()
                if result is not None:
                    st.subheader("Output")
                    st.json(result)
                    st.text(result["Contents"])

if __name__ == "__main__":
    main()
