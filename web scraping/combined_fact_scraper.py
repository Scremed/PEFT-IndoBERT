# Scraper Title + Date (Detik, Kompas, Tempo), Jan 2024 - September 2026.
import argparse
import json
import random
import re
import sys
import time
import unicodedata
import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

# Configuration
START = date(2024, 1, 1) # Start: 1 January 2024
END = date(2026, 9, 23)  # End: 23 September 2026
TARGET_TOTAL = 6394         
PER_DAY = 4                     
MAX_PAGE_FALLBACK = 5         
SEED = 42
DELAY = (3.5, 10.8)         
SEM_THRESHOLD = 0.78      
SEM_WINDOW_DAYS = 3   
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CACHE = Path("raw_cache.jsonl")
OUT = Path("datasets/raw/fact_news_dataset.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 research-scraper"
}

# URL daily news index 
SOURCES = {
    "detik": {
        "url": lambda d, p: f"https://news.detik.com/indeks?page={p}&date={d.strftime('%m/%d/%Y')}",
        "selectors": ["article h3.media__title a", "h3.media__title a", "article h2 a", "article h3 a"],
    },
    "kompas": {
        "url": lambda d, p: f"https://indeks.kompas.com/?site=all&date={d.isoformat()}&page={p}",
        "selectors": [".articleItem .articleTitle", "h2.articleTitle", ".articleItem h2", ".articleItem h3"],
    },
    "liputan6": {
        "url": lambda d, p: (
            f"https://www.liputan6.com/news/indeks/{d.strftime('%Y/%m/%d')}"
            f"?end={(d + timedelta(days=1)).isoformat()}&start={d.isoformat()}&page={p}"
        ),
        "selectors": ["article h4 a", "h4.articles--iridescent-list--text-item__title a",
                      "article h4", "article h3 a", "article h2 a"],
    },
}

# Util
_robots = {}


def allowed(url):
    host = urlparse(url)
    base = f"{host.scheme}://{host.netloc}"
    if base not in _robots:
        rp = RobotFileParser()
        rp.set_url(base + "/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None
        _robots[base] = rp
    rp = _robots[base]
    return True if rp is None else rp.can_fetch(HEADERS["User-Agent"], url)


def clean(t):
    t = unicodedata.normalize("NFKC", t)
    return re.sub(r"\s+", " ", t).strip()


def norm(t):
    t = unicodedata.normalize("NFKC", t).lower()
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def daterange(a, b):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def fetch_titles(source, d, session, page=1):
    """Return (titles, max_page) or None if failed/blocked."""
    cfg = SOURCES[source]
    url = cfg["url"](d, page)
    if not allowed(url):
        print(f"[robots.txt melarang] {url}", file=sys.stderr)
        return None
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, "html.parser")
                pages = [int(m) for a in soup.select("a[href]")
                         for m in re.findall(r"[?&]page=(\d+)", a["href"])]
                max_page = max(pages) if pages else MAX_PAGE_FALLBACK
                for sel in cfg["selectors"]:
                    els = soup.select(sel)
                    titles = [clean(e.get_text(" ")) for e in els]
                    titles = [t for t in titles if len(t) >= 20]
                    if titles:
                        return list(dict.fromkeys(titles)), max_page
                return [], max_page
            if r.status_code in (403, 429, 503):
                time.sleep(10 * (attempt + 1))
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
    return None 

# Test
def cmd_test(_):
    s = requests.Session()
    d = END - timedelta(days=3)
    for src in SOURCES:
        res = fetch_titles(src, d, s)
        titles = None if res is None else res[0]
        info = "" if res is None else f", halaman terdeteksi: {res[1]}"
        print(f"\n== {src} ({d}) -> {None if titles is None else len(titles)} judul{info}")
        for t in (titles or [])[:5]:
            print("  -", t)
        time.sleep(random.uniform(*DELAY))
    print("\nJika ada sumber yang 0/None, sesuaikan URL/selector di SOURCES.")


# Scrape
def load_cache():
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["source"], r["date"])] = r["titles"]
    return done


def cmd_scrape(_):
    rnd = random.Random(SEED)
    done = load_cache()
    s = requests.Session()
    days = list(daterange(START, END))
    total = len(days) * len(SOURCES)
    n = 0
    with CACHE.open("a", encoding="utf-8") as f:
        for d in days:
            for src in SOURCES:
                n += 1
                key = (src, d.isoformat())
                if key in done:
                    continue
                res = fetch_titles(src, d, s)
                if res is None:
                    continue
                t1, max_page = res
                time.sleep(random.uniform(*DELAY))
                t2 = []
                if max_page >= 2:
                    res2 = fetch_titles(src, d, s, rnd.randint(2, max_page))
                    t2 = res2[0] if res2 else []
                half = PER_DAY // 2
                pick = rnd.sample(t1, min(half, len(t1)))
                pool = t2 or [t for t in t1 if t not in pick]
                pick += rnd.sample(pool, min(PER_DAY - len(pick), len(pool)))
                f.write(json.dumps({"source": src, "date": d.isoformat(), "titles": pick},
                                   ensure_ascii=False) + "\n")
                f.flush()
                time.sleep(random.uniform(*DELAY))
            if d.day == 1 or d == END:
                print(f"progress {n}/{total} (sampai {d})")


# Build
def cmd_build(_):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    rnd = random.Random(SEED)
    rows, seen = [], set()
    for (src, d), titles in load_cache().items():
        for t in titles:
            k = norm(t)
            if k in seen:              
                continue
            seen.add(k)
            rows.append({"source": src, "date": date.fromisoformat(d), "title": t})
    print(f"data mentah unik (exact): {len(rows)}")

    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode([r["title"] for r in rows], normalize_embeddings=True,
                       batch_size=64, show_progress_bar=True)
    order = list(range(len(rows)))
    rnd.shuffle(order)
    kept, by_date = [], defaultdict(list)
    for i in order:
        d = rows[i]["date"]
        cand = [j for k in range(-SEM_WINDOW_DAYS, SEM_WINDOW_DAYS + 1)
                for j in by_date.get(d + timedelta(days=k), [])]
        if cand and float((emb[cand] @ emb[i]).max()) >= SEM_THRESHOLD:
            continue
        kept.append(i)
        by_date[d].append(i)
    print(f"setelah dedup semantik: {len(kept)}")

    srcs = list(SOURCES)
    base, extra = divmod(TARGET_TOTAL, len(srcs))
    final = []
    for idx, src in enumerate(srcs):
        target = base + (1 if idx < extra else 0)
        months = defaultdict(list)
        for i in kept:
            if rows[i]["source"] == src:
                months[rows[i]["date"].strftime("%Y-%m")].append(i)
        for m in months.values():
            rnd.shuffle(m)
        picked, keys = [], sorted(months)
        while len(picked) < target and any(months[k] for k in keys):
            for k in keys:
                if months[k] and len(picked) < target:
                    picked.append(months[k].pop())
        print(f"{src}: {len(picked)}/{target}")
        final += picked

    final.sort(key=lambda i: (rows[i]["date"], rows[i]["source"]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["title", "date", "source"])
        for i in final:
            r = rows[i]
            w.writerow([r["title"], r["date"].isoformat(), r["source"]])
    print(f"tersimpan: {OUT} ({len(final)} baris)")
    if len(final) < TARGET_TOTAL:
        print(f"KURANG {TARGET_TOTAL - len(final)} baris -> naikkan PER_DAY lalu jalankan scrape ulang, "
              f"atau turunkan SEM_THRESHOLD.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "scrape", "build"])
    a = ap.parse_args()
    {"test": cmd_test, "scrape": cmd_scrape, "build": cmd_build}[a.cmd](a)