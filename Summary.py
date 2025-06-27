import argparse
import logging
import os
import re
import tempfile

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from pytube import YouTube
import whisper

logging.basicConfig(level=logging.INFO)

def get_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id: str, with_timestamps: bool = False) -> str | None:
    """Fetch the transcript for a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if with_timestamps:
            return " ".join([f"[{entry['start']:.1f}s] {entry['text']}" for entry in transcript])
        else:
            return " ".join([entry['text'] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        logging.error(f"Failed to fetch transcript: {e}")
        return None

def download_audio(youtube_url: str, filename: str) -> str:
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    if not audio_stream:
        raise RuntimeError("No audio stream found for this video.")
    audio_stream.download(filename=filename)
    return filename

def transcribe_audio_whisper(audio_path: str, with_timestamps=False) -> str:
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=False)
    if with_timestamps:
        return " ".join([f"[{seg['start']:.1f}s] {seg['text'].strip()}" for seg in result["segments"]])
    else:
        return result["text"]

def summarize_text(text: str, max_length=512) -> str:
    import torch
    model_name = "google/bigbird-pegasus-large-arxiv"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    max_chunk_tokens = 3500
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"][0]
    summaries = []
    for i in range(0, len(input_ids), max_chunk_tokens):
        chunk_ids = input_ids[i:i+max_chunk_tokens]
        if len(chunk_ids) < 10:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        try:
            summary = summarizer(chunk_text, max_length=max_length, min_length=40, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            logging.warning(f"Skipped chunk due to error: {e}")
    combined_summary = " ".join(summaries)
    if len(combined_summary.split()) > max_length * 2:
        try:
            summary = summarizer(combined_summary, max_length=max_length, min_length=40, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            logging.warning(f"Second pass summarization failed: {e}")
            return combined_summary
    else:
        return combined_summary

def summarize_youtube_video(url: str, with_timestamps=False) -> str:
    video_id = get_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    transcript = fetch_transcript(video_id, with_timestamps=with_timestamps)
    if transcript:
        logging.info("Transcript found. Summarizing transcript...")
        return summarize_text(transcript)
    else:
        logging.info("Transcript not available. Downloading audio and transcribing...")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_audio:
            try:
                download_audio(url, tmp_audio.name)
                text = transcribe_audio_whisper(tmp_audio.name, with_timestamps=with_timestamps)
                return summarize_text(text)
            finally:
                os.remove(tmp_audio.name)

def main():
    parser = argparse.ArgumentParser(description="Summarize YouTube videos under 4 minutes.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps in summaries")
    parser.add_argument("--output", help="Save summary to file")
    args = parser.parse_args()

    try:
        summary = summarize_youtube_video(args.url, with_timestamps=args.timestamps)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"Summary saved to {args.output}")
        else:
            print("\nSummary:\n", summary)
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == "__main__":
    main()
