import requests
import re
import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pytube import YouTube
import whisper

def get_video_id(url):
    # Extract video ID from YouTube URL
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id, with_timestamps=False):
    # Fetch transcript using YouTubeTranscriptApi
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if with_timestamps:
            text = " ".join([f"[{entry['start']:.1f}s] {entry['text']}" for entry in transcript])
        else:
            text = " ".join([entry['text'] for entry in transcript])
        return text
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        print(f"Failed to fetch transcript: {e}")
        return None

def download_audio(youtube_url, filename="audio.mp4"):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(filename=filename)
    return filename

def transcribe_audio_whisper(audio_path, with_timestamps=False):
    model = whisper.load_model("base")  # You can use "small" or "medium" for better accuracy
    result = model.transcribe(audio_path, verbose=False)
    if with_timestamps:
        # Include timestamps for each segment
        text = " ".join([f"[{seg['start']:.1f}s] {seg['text'].strip()}" for seg in result["segments"]])
    else:
        text = result["text"]
    return text

def summarize_text(text, max_length=512):  # Increased from 256 to 512
    import torch
    model_name = "google/bigbird-pegasus-large-arxiv"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    max_chunk_tokens = 3500  # Stay well below 4096
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    summaries = []

    # Split input_ids into safe-sized chunks
    for i in range(0, len(input_ids), max_chunk_tokens):
        chunk_ids = input_ids[i:i+max_chunk_tokens]
        # Truncate chunk if it exceeds max_chunk_tokens
        if len(chunk_ids) > max_chunk_tokens:
            chunk_ids = chunk_ids[:max_chunk_tokens]
        if len(chunk_ids) < 10:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        try:
            summary = summarizer(chunk_text, max_length=max_length, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            print(f"Warning: Skipped a chunk due to error: {e}")

    combined_summary = " ".join(summaries)
    # Second pass if still too long
    if len(combined_summary.split()) > max_length * 2:
        try:
            summary = summarizer(combined_summary, max_length=max_length, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Warning: Second pass summarization failed: {e}")
            return combined_summary
    else:
        return combined_summary

def summarize_youtube_video(url, with_timestamps=False):
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    transcript = fetch_transcript(video_id, with_timestamps=with_timestamps)
    if transcript:
        print("Transcript found. Summarizing transcript...")
        return summarize_text(transcript)
    else:
        print("Transcript not available. Downloading audio and transcribing...")
        audio_file = "audio.mp4"
        download_audio(url, audio_file)
        text = transcribe_audio_whisper(audio_file, with_timestamps=with_timestamps)
        os.remove(audio_file)  # Clean up
        return summarize_text(text)

def is_transcript_available(video_id, languages=['en']):
    """
    Returns True if a transcript is available for the given video_id, else False.
    """
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcripts:
            if transcript.language_code in languages or transcript.is_generated:
                return True
        return False
    except (TranscriptsDisabled, NoTranscriptFound):
        return False
    except Exception as e:
        print(f"Error checking transcript: {e}")
        return False

# Example usage before summarizing:
if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    # Set with_timestamps=True if you want timestamps in the context
    try:
        summary = summarize_youtube_video(youtube_url, with_timestamps=True)
        print("\nSummary:\n", summary)
    except Exception as e:
        print("Error:", e)
        print("\nSummary:\n", summary)
    except Exception as e:
        print("Error:", e)
