import os
import json
import datetime
import openai

from flask import Flask, render_template, request
from llama_index import LLMPredictor, GPTVectorStoreIndex, ServiceContext, SimpleDirectoryReader
from langchain.chat_models import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
from html2image import Html2Image

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = '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'

# Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "data"
app.secret_key = os.urandom(12).hex()

# Paths
doc_path = './data/'
transcript_file = './data/transcript.json'
index_file = 'index.json'
youtube_img = 'thumbnail.png'

def summarize_video(video_id):
    """Fetches transcript, generates summary, and extracts sections."""
    
    # Fetch YouTube transcript
    srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    formatter = JSONFormatter()
    json_formatted = formatter.format_transcript(srt)
    
    # Save transcript
    with open(transcript_file, 'w') as f: 
        f.write(json_formatted)

    # Generate thumbnail
    hti = Html2Image(output_path=app.config["UPLOAD_FOLDER"])
    hti.screenshot(url=f"https://www.youtube.com/watch?v={video_id}", save_as=youtube_img)

    # Load documents into LlamaIndex
    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()

    # Define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=500))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(index_file)
    query_engine = index.as_query_engine()

    section_texts = ''
    section_start_s = 0
    section_response = ''

    with open(transcript_file, 'r') as f:
        transcript = json.load(f)

    start_text = transcript[0]["text"]

    for d in transcript:
        if d["start"] <= (section_start_s + 300) and transcript.index(d) != len(transcript) - 1:
            section_texts += ' ' + d["text"]
        else:
            end_text = d["text"]
            prompt = f"Summarize this section from \"{start_text}\" to \"{end_text}\", limit to 100 words."
            response = query_engine.query(prompt)
            start_time = str(datetime.timedelta(seconds=section_start_s))
            end_time = str(datetime.timedelta(seconds=int(d['start'])))
            section_start_s += 300
            start_text = d["text"]
            section_texts = ''
            section_response += f"**{start_time} - {end_time}:**\n{response}\n\n"

    # Generate overall summary
    final_prompt = "Summarize this video based on its sections: " + section_response
    final_summary = query_engine.query(final_prompt)

    return final_summary, section_response

@app.route("/", methods=["GET", "POST"])
def home():
    """Handles YouTube video processing and summary generation."""
    summary = ""
    section_details = ""
    imgPath = ""

    if request.method == "POST":
        link = request.form.get("URL")

        if link:
            video_id = link.split("v=")[1][:11]
            summary, section_details = summarize_video(video_id)
            imgPath = os.path.join(app.config["UPLOAD_FOLDER"], youtube_img)
        else:
            summary = "Invalid YouTube link."

    return render_template("home.html", summary=summary, section_details=section_details, image_path=imgPath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)
