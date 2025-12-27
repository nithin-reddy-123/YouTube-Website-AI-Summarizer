import streamlit as st
import ssl
import validators
import re

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

def get_youtube_transcript_docs(url: str):
    def extract_video_id(youtube_url):
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
        return match.group(1) if match else None

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = YouTubeTranscriptApi().fetch(video_id)

    full_text = " ".join(snippet.text for snippet in transcript)

    return [Document(page_content=full_text)]


# Streamlit UI
st.title("YT / Website Summarizer")

# Your Groq API Key stored in secrets file
api_key = st.secrets["api"]["api_key"]

llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = PromptTemplate(
    template="Provide a summary of the following content in 300 words:\n\n{text}",
    input_variables=["text"]
)

chain = prompt | llm | StrOutputParser()

url = st.text_input("Enter YouTube or Website URL")

if st.button("Summarize"):
    if not url.strip():
        st.error("⚠️ Please provide a URL.")
    elif not validators.url(url):
        st.error("❌ Please enter a valid URL.")
    else:
        try:
            with st.spinner("⏳ Fetching and summarizing..."):
                if "youtube.com" in url or "youtu.be" in url:
                    docs = get_youtube_transcript_docs(url)
                else:
                    loader = UnstructuredURLLoader(urls=[url])
                    docs = loader.load()

                summary = chain.invoke({"text": docs[0].page_content})

                st.success("✅ Summary Generated:")
                st.write(summary)

        except Exception as e:
            st.error(f"❌ Error: {e}")
