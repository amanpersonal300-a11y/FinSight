"""Run once: downloads latest 10-K for each ticker and loads it into the vector DB."""
import os, re, sys, glob
from dotenv import load_dotenv
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup
import chromadb

load_dotenv()
TICKERS = sys.argv[1:] or ["NVDA", "AMD", "INTC", "AAPL", "MSFT"]
dl = Downloader(os.getenv("SEC_NAME", "Student"), os.getenv("SEC_EMAIL", "student@example.com"), "data/sec")
col = chromadb.PersistentClient(path="data/chroma_db").get_or_create_collection("filings")

def chunk(words, size=300, overlap=50):
    for i in range(0, len(words), size - overlap):
        yield " ".join(words[i:i + size])

for t in TICKERS:
    print(f"Downloading {t} 10-K ...")
    try:
        dl.get("10-K", t, limit=1)
    except Exception as e:
        print("  failed:", e); continue
    files = glob.glob(f"data/sec/sec-edgar-filings/{t}/10-K/*/full-submission.txt")
    if not files:
        print("  no file found"); continue
    path = files[0]
    accession = path.split(os.sep)[-2]
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"<TYPE>10-K.*?<TEXT>(.*?)</TEXT>", raw, re.S)
    html = m.group(1) if m else raw
    text = BeautifulSoup(html, "html.parser").get_text(" ")
    words = text.split()
    chunks = list(chunk(words))
    print(f"  {len(chunks)} chunks, embedding ...")
    for s in range(0, len(chunks), 100):
        batch = chunks[s:s + 100]
        col.upsert(
            ids=[f"{t}-{accession}-{s + j}" for j in range(len(batch))],
            documents=batch,
            metadatas=[{"ticker": t, "form": "10-K", "accession": accession}] * len(batch))
print("Done. Total chunks in DB:", col.count())
