import requests
import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

def get_video_id(url):
    # Extract video ID from YouTube URL
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id):
    # Fetch transcript using YouTubeTranscriptApi
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry['text'] for entry in transcript])
    return text

def summarize_text(text, max_length=150):
    # Use HuggingFace transformers for summarization
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_length, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def summarize_youtube_video(url):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    transcript = fetch_transcript(video_id)
    # Transformers models have a max token limit, so chunk if needed
    if len(transcript.split()) > 1000:
        transcript = " ".join(transcript.split()[:1000])
    summary = summarize_text(transcript)
    return summary

if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    try:
        summary = summarize_youtube_video(youtube_url)
        print("\nSummary:\n", summary)
    except Exception as e:
        print("Error:", e)