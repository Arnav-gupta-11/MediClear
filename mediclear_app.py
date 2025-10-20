import io
import json
import os
import re
from typing import Tuple

import requests
import streamlit as st
from streamlit.components.v1 import html as st_html

try:
    from streamlit_option_menu import option_menu
except Exception:
    option_menu = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip() or "AIzaSyDP24UgGKpXXs80SOaeHq3ouP-aCB4nCHI"
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "").strip() or "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash").strip()

st.set_page_config(page_title="MediClear — Medical Docs, Simplified", layout="wide")


def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return "[PDF parsing unavailable: install pdfplumber]"
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        return f"[Error parsing PDF: {e}]"
    return "\n\n".join(text_parts).strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    if docx is None:
        return "[DOCX parsing unavailable: install python-docx]"
    tmp_path = "/tmp/mediclear_tmp.docx"
    try:
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)
        doc = docx.Document(tmp_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return "\n\n".join(paragraphs).strip()
    except Exception as e:
        return f"[Error parsing DOCX: {e}]"

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return str(file_bytes)

def extract_text_from_image(file_bytes: bytes) -> str:
    if pytesseract is None or Image is None:
        return "[OCR unavailable: install pillow and pytesseract]"
    try:
        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"[OCR error: {e}]"

def extract_text(uploaded_file) -> Tuple[str, str]:
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(raw), "PDF"
    if filename.endswith(".docx"):
        return extract_text_from_docx(raw), "DOCX"
    if filename.endswith(".txt"):
        return extract_text_from_txt(raw), "TXT"
    if any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tiff"]):
        return extract_text_from_image(raw), "IMAGE"
    return extract_text_from_txt(raw), "UNKNOWN"

def _parse_gemini_response(resp_json):
    if not resp_json:
        return None

    try:
        candidates = resp_json.get("candidates") or resp_json.get("candidate")
        if isinstance(candidates, list) and candidates:
            first = candidates[0]
            content = first.get("content") or first.get("message") or first
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list) and parts:
                    p0 = parts[0]
                    if isinstance(p0, dict) and "text" in p0:
                        return p0["text"]
                    if isinstance(p0, str):
                        return p0
            if "text" in first:
                return first["text"]
    except Exception:
        pass

    try:
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            if "text" in choices[0]:
                return choices[0]["text"]
            msg = choices[0].get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                    return content[0]["text"]
    except Exception:
        pass

    try:
        output = resp_json.get("output")
        if isinstance(output, dict) and "text" in output:
            return output["text"]
    except Exception:
        pass

    try:
        return json.dumps(resp_json, indent=2)
    except Exception:
        return str(resp_json)

def call_gemini(prompt: str, system: str = "", temperature: float = 0.0, max_tokens: int = 800) -> str:
    if not GEMINI_API_KEY:
        return (
            "[Gemini API not configured. Set GEMINI_API_KEY environment variable or edit the script.]\n\n"
            f"Prompt preview:\n\n{prompt[:2000]}..."
        )

    url = GEMINI_API_URL if GEMINI_API_URL else f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    contents = []
    system_instruction = None
    if system:
        system_instruction = {"parts": [{"text": system}]}
    contents.append({"parts": [{"text": prompt}]})

    body = {
        "contents": contents,
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
            "candidateCount": 1,
        },
    }
    if system_instruction:
        body["systemInstruction"] = system_instruction

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
    except Exception as e:
        return f"[Error contacting Gemini API: network error: {e}]\n\nPrompt preview:\n\n{prompt[:2000]}..."

    if resp.status_code >= 400:
        resp_text = resp.text
        try:
            resp_json = resp.json()
            resp_text = json.dumps(resp_json, indent=2)
        except Exception:
            pass
        return f"[Gemini API returned HTTP {resp.status_code}]\n\n{resp_text}\n\nPrompt preview:\n\n{prompt[:2000]}..."

    try:
        data = resp.json()
    except Exception as e:
        return f"[Error parsing Gemini response JSON: {e}]\n\nRaw response:\n\n{resp.text}\n\nPrompt preview:\n\n{prompt[:2000]}..."

    parsed = _parse_gemini_response(data)
    if parsed is None:
        try:
            return json.dumps(data, indent=2)
        except Exception:
            return str(data)
    return parsed

DOC_TYPE_CLASSIFIER_PROMPT = """
You are a classifier that identifies the type of medical document. Output exactly one label from:
Lab Results, Radiology Report, Pathology Report, Discharge Summary, Prescription, Insurance EOB, Referral Letter, Surgical Note, Other.

Text:
\"\"\"
{doc_text}
\"\"\"
"""

EXPLAINER_PROMPT = """
You are MediClear — an AI that explains medical documents in plain language for patients.  
Follow this exact output format:  

1. **Document Type:** One short line identifying what the document is.  
2. **Summary:** 3–6 bullet points with the most important information only. Use simple language.  
3. **Definitions:** For each medical term or abbreviation in the document, give a 1-sentence plain-English definition.  
4. **Follow-Up Questions for a Clinician:** Up to 3 polite, non-diagnostic questions a patient might ask.  
5. **Notable Numeric Values:** List any numbers (lab results, vitals, measurements) and please provide their significance to the particular document, and please include what they are for. (do not guess).  
6. **Disclaimer:** Always end with — *"This is for educational purposes only and is not medical advice."*

Rules:  
- Never give a diagnosis or treatment recommendation.  
- Avoid speculation about conditions.  
- Keep all explanations patient-friendly and easy to read.

Document:
\"\"\"
{doc_text}
\"\"\"
"""

def short_preview(s: str, n=800):
    return (s[:n] + "...") if len(s) > n else s

def strip_long_whitespace(s: str) -> str:
    return re.sub(r"\n\s+\n", "\n\n", s).strip()



def home_page():
    st.markdown(
        """
        <h1 style="
            text-align: center;
            font-weight: 600;
            font-size: 4em;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            color: #1a1a1a;
            font-family: 'Open Sans', Arial, sans-serif;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            padding-top: -0.1rem;
        ">
            MediClear
        </h1>
        """,
        unsafe_allow_html=True
    )

    
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem;">
            <h2 style="margin: 0 0 1rem 0; color: white;">Simplifying Medical Information for Everyone</h2>
            <p style="font-size: 1.1rem; margin: 0; opacity: 0.95;">
            MediClear transforms complex medical documents into easy-to-understand summaries, 
            helping you better understand your health and making informed decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        border-left: 4px solid #0d6efd; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; font-weight: bold; color: #0d6efd;">12%</div>
                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">
                    Have proficient health literacy
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        border-left: 4px solid #dc3545; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; font-weight: bold; color: #dc3545;">36%</div>
                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">
                    Have basic or below-basic literacy
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; 
                        border-left: 4px solid #fd7e14; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                <div style="font-size: 2.5rem; font-weight: bold; color: #fd7e14;">88%</div>
                <div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.5rem;">
                    Struggle in at least one area
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.info("**What this means:** Most adults need plain-language summaries and clear instructions, especially for medications and treatment plans.")
    
    left, right = st.columns([1, 1])
    
    with left:
        st.markdown("### Why Health Literacy Matters")
        st.markdown("""
            Health literacy affects how well people can:
            - Understand prescription instructions
            - Follow medication dosing schedules
            - Recognize warning signs and side effects
            - Make informed healthcare decisions
            - Navigate the healthcare system
        """)
        
        st.warning("""
            **Studies show:**
            - High rates of misunderstanding prescription labels
            - Patients frequently misread dosing instructions
            - Auxiliary warnings are often ignored or misunderstood
            - Confusion leads to medication errors and poor outcomes
        """)
    
    with right:
        st.markdown("### How MediClear Helps")
        st.markdown("""
            MediClear addresses these challenges by:
            - **Simplifying complex documents** into plain language
            - **Creating clear summaries** of key information
            - **Providing definitions** for medical terms
            - **Suggesting questions** to ask your healthcare provider
            - **Highlighting important** medication instructions
        """)
        
        st.success("""
            **The Result:**
            Clear communication reduces patient confusion and supports safer, 
            more effective healthcare for everyone.
        """)
    st.markdown("---")
def upload_page():
    
    st.markdown(
        "Upload any **medical document** (lab results, radiology report, discharge summary, prescription, EOB). "
        "MediClear will extract text and produce a patient-friendly summary, glossary, and suggested follow-ups."
    )

    uploaded_file = st.file_uploader("Upload a medical document (PDF, DOCX, TXT, JPG/PNG)", type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'tiff'])
    if uploaded_file is None:
        return

    with st.spinner("Extracting text..."):
        extracted, ftype = extract_text(uploaded_file)
        extracted = strip_long_whitespace(extracted)

    st.write("Detecting document type...")
    classifier_prompt = DOC_TYPE_CLASSIFIER_PROMPT.format(doc_text=extracted[:4000])
    doc_type_resp = call_gemini(classifier_prompt, system="You are a short, strict classifier. Return exactly one label.")
    st.markdown(f"**Detected type (raw model output):** `{short_preview(doc_type_resp, 400)}`")

    doc_type_options = ["Lab Results", "Radiology Report", "Pathology Report", "Discharge Summary",
                        "Prescription", "Insurance EOB", "Referral Letter", "Surgical Note", "Other"]
    try:
        index = doc_type_options.index(doc_type_resp.strip())
    except Exception:
        index = 0
    user_selected_type = st.selectbox("If detection is wrong, choose the correct type:", options=doc_type_options, index=index)

    st.write("Generating patient-friendly explanation...")
    expl_prompt = EXPLAINER_PROMPT.format(doc_text=extracted)
    final_output = call_gemini(
        expl_prompt,
        system="You are a helpful medical-document explainer. Keep language simple and patient-friendly.",
        temperature=0.0,
        max_tokens=4096,
    )

    st.subheader("MediClear Explanation")
    if final_output.startswith("[Gemini API not configured"):
        st.warning(final_output)
        st.info("Replace GEMINI_API_KEY at the top of the script or set it as an environment variable.")
    elif final_output.startswith("[Gemini API returned HTTP") or final_output.startswith("[Error"):
        st.error(final_output)
        st.info("Copy the full error message here if you want help debugging.")
    else:
        st.markdown(final_output)

    st.download_button("Download explanation as .txt", data=final_output, file_name="mediclear_explanation.txt")
    

    

def sources():
    with st.expander("View Sources & Research"):
        st.markdown("### Health Literacy Statistics Sources")

        st.markdown(
            "**Proficient Health Literacy (12%)**  \n"
            "> Only about 12% of U.S. adults have 'proficient' health literacy skills...  \n"
            "*Source: AHRQ / HHS / NIH — Health Literacy: Report from the 2003 NAAL*\n\n"
            "**Basic or Below-Basic Literacy (36%)**  \n"
            "> Approximately 36% of U.S. adults have 'basic' or 'below-basic' health literacy...  \n"
            "*Source: NCBI / Journal of General Internal Medicine*\n\n"
            "**Population Struggles (88%)**  \n"
            "> Nearly 9 out of 10 adults in the U.S. struggle with health literacy...  \n"
            "*Source: NNLM*"
        )

        st.markdown(
            "### Additional References & Evidence\n"
            "Key sources referenced or inspiring the app content:\n"
            "- **AHRQ / HHS / NIH** — Health Literacy: Report from the 2003 National Assessment of Adult Literacy (NAAL)\n"
            "- **NCBI / Journal of General Internal Medicine** — Studies on basic and below-basic literacy estimates\n"
            "- **NNLM** — Commentary on population-level health literacy\n"
            "- **AAFP / American Family Physician** — Research on medication label misunderstandings"
        )
        
    with st.expander("Technical Workflow"):
        st.markdown("""
        MediClear processes medical documents and produces patient-friendly explanations through the following steps:

        1. **File Upload & Type Detection**
        - Users upload medical documents (`PDF`, `DOCX`, `TXT`, or image formats `JPG/PNG/TIFF`).
        - The system reads the file bytes and determines the file type.
        - For images, OCR (`pytesseract`) is applied to extract text.

        2. **Text Extraction**
        - **PDF:** `pdfplumber` extracts text from all pages.
        - **DOCX:** `python-docx` reads paragraphs and merges them.
        - **TXT:** Decoded directly as UTF-8.
        - **Images:** OCR converts visuals to text.
        - Any errors are captured and returned as warnings to the user.

        3. **Document Classification**
        - A generative model prompt classifies the text into one of:
            `Lab Results`, `Radiology Report`, `Pathology Report`, `Discharge Summary`, `Prescription`, `Insurance EOB`, `Referral Letter`, `Surgical Note`, `Other`.
        - The model output is displayed, and the user can manually override if needed.

        4. **Patient-Friendly Explanation Generation**
        - The extracted text is fed into the **Explainer Prompt**:
            - Produces document type confirmation.
            - Summarizes key points (3–6 bullets).
            - Provides definitions for medical terms.
            - Suggests 0–3 follow-up questions to ask a clinician.
            - Highlights numeric values with context.
            - Adds a disclaimer for educational purposes.
        - The Google Gemini API handles the text generation, ensuring language is simple and patient-friendly.

        5. **Display & Download**
        - The explanation is displayed in the Streamlit app.
        - Users can download the explanation as a `.txt` file for personal reference.

        6. **Error Handling & Fallbacks**
        - If any module (`pdfplumber`, `python-docx`, `pytesseract`) is unavailable, appropriate messages are shown.
        - Google Gemini API errors are caught and displayed with prompt previews.
        - Users are informed if manual intervention is needed.

        **Technical Stack:**
        - Streamlit for UI/UX
        - `pdfplumber`, `python-docx`, `pytesseract` for document parsing
        - Google Gemini generative model API for classification & explanation
        - Python 3.10+ with standard libraries (`io`, `json`, `os`, `re`, `requests`)
        """)
    
    
    
    
    
    
    
st.markdown("---")




def main():
    home_page()
    upload_page()
    sources()
if __name__ == "__main__":
    main()