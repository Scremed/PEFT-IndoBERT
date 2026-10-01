import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import random
import os
import json
from datetime import datetime
from tqdm import tqdm


BASE_URL = "https://turnbackhoax.id"
ARTICLES_URL = f"{BASE_URL}/articles"

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime.now()

OUTPUT_FILE = "datasets/raw/turnbackhoax_2024_2026_raw.csv"
CHECKPOINT_FILE = "datasets/raw/turnbackhoax_checkpoint.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/139.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# REQUEST
# ============================================================

def get_page(url, retries=3):

    for attempt in range(retries):

        try:
            response = session.get(
                url,
                timeout=30
            )

            if response.status_code == 200:
                return response

            print(
                f"HTTP {response.status_code}: {url}"
            )

        except requests.RequestException as e:

            print(
                f"Request error: {e}"
            )

        time.sleep(2)

    return None


# ============================================================
# EXTRACT LABEL
# ============================================================

def extract_label(title):

    if not title:
        return None

    match = re.match(
        r"^\s*\[([^\]]+)\]",
        title
    )

    if match:
        raw_label = match.group(1).strip().upper()

        if raw_label in {"BENAR", "FAKTA"}:
            return 0

        if raw_label in {"SALAH", "PENIPUAN", "SESAT", "MANIPULASI"}:
            return 1

    return None


# ============================================================
# EXTRACT ARTICLES FROM PAGINATION PAGE
# ============================================================

def get_article_links(page):

    url = f"{ARTICLES_URL}?page={page}"

    response = get_page(url)

    if response is None:
        return []

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    articles = []

    cards = soup.select(
        ".news-card-h-alt"
    )

    for card in cards:

        # ----------------------------------------------------
        # ARTICLE LINK
        # ----------------------------------------------------

        link = card.find(
            "a",
            href=re.compile(
                r"/articles/\d+-"
            )
        )

        if not link:
            continue

        article_url = link.get("href")

        # Make URL absolute
        if article_url.startswith("/"):
            article_url = BASE_URL + article_url

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        # The visible h2 spans are truncated for the responsive layout.
        # The image alt attribute contains the complete article title.
        image = card.find("img", alt=True)
        title = image.get("alt", "").strip() if image else None

        if not title:
            title_tag = card.find("h2")
            title = title_tag.get_text(" ", strip=True) if title_tag else None

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        date_tag = card.find(
            "span",
            class_="text-light-black"
        )

        if date_tag:

            date_text = date_tag.get_text(
                " ",
                strip=True
            )

        else:

            date_text = None

        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = extract_label(title)

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        articles.append({
            "berita": title,
            "tanggal_publikasi": date_text,
            "label": label
        })

    return articles


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint():

    if not os.path.exists(CHECKPOINT_FILE):
        return 1

    try:

        with open(
            CHECKPOINT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            checkpoint = json.load(f)

        return checkpoint.get(
            "next_page",
            1
        )

    except Exception:

        print(
            "Could not read checkpoint. "
            "Starting from page 1."
        )

        return 1


def save_checkpoint(next_page):

    os.makedirs(
        os.path.dirname(CHECKPOINT_FILE),
        exist_ok=True
    )

    checkpoint = {
        "next_page": next_page,
        "updated_at": datetime.now().isoformat()
    }

    with open(
        CHECKPOINT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            checkpoint,
            f,
            indent=4
        )


# ============================================================
# SAVE DATA
# ============================================================

def save_articles(articles):

    if not articles:
        return

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df = pd.DataFrame(
        articles
    )

    # Convert date
    df["tanggal_publikasi"] = pd.to_datetime(
        df["tanggal_publikasi"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Keep only requested date range
    df = df[
        (df["tanggal_publikasi"] >= START_DATE)
        &
        (df["tanggal_publikasi"] <= END_DATE)
    ]

    # Save using original date format
    df["tanggal_publikasi"] = df[
        "tanggal_publikasi"
    ].dt.strftime("%d/%m/%Y")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Saved {len(df)} articles → {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD EXISTING DATA
    # --------------------------------------------------------

    all_articles = []

    if os.path.exists(OUTPUT_FILE):

        print(
            f"Existing dataset found: {OUTPUT_FILE}"
        )

        existing_df = pd.read_csv(
            OUTPUT_FILE
        )

        all_articles = (
            existing_df
            .to_dict("records")
        )

        print(
            f"Existing articles: "
            f"{len(all_articles)}"
        )

    # --------------------------------------------------------
    # LOAD CHECKPOINT
    # --------------------------------------------------------

    page = load_checkpoint()

    print("=" * 60)
    print("TURNBACKHOAX SCRAPER")
    print("=" * 60)

    print(
        f"Starting from page: {page}"
    )

    if page > 1:

        print(
            "Resuming previous scraping session..."
        )

    # --------------------------------------------------------
    # SCRAPE PAGINATION
    # --------------------------------------------------------

    while True:

        print()
        print(
            f"Scraping listing page {page}..."
        )

        articles = get_article_links(
            page
        )

        if not articles:

            print(
                "No articles found."
            )

            print(
                "The page may have failed. "
                "Checkpoint was NOT advanced."
            )

            break

        print(
            f"Found {len(articles)} articles"
        )

        all_articles.extend(
            articles
        )

        # ----------------------------------------------------
        # CHECK OLDEST ARTICLE
        # ----------------------------------------------------

        dates = []

        for article in articles:

            try:

                date = datetime.strptime(
                    article["tanggal_publikasi"],
                    "%d/%m/%Y"
                )

                dates.append(date)

            except (
                ValueError,
                TypeError
            ):
                pass

        reached_start = False

        if dates:

            oldest_date = min(dates)

            print(
                "Oldest article: "
                f"{oldest_date.strftime('%d/%m/%Y')}"
            )

            if oldest_date < START_DATE:

                print()
                print(
                    "Reached articles older "
                    "than January 2024."
                )

                reached_start = True

        # ----------------------------------------------------
        # SAVE AFTER EVERY PAGE
        # ----------------------------------------------------

        save_articles(
            all_articles
        )

        # ----------------------------------------------------
        # STOP CONDITION
        # ----------------------------------------------------

        if reached_start:

            print(
                "Scraping completed."
            )

            # Mark scraper as finished
            save_checkpoint(
                page
            )

            break

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        save_checkpoint(
            page + 1
        )

        print(
            f"Checkpoint saved: "
            f"next page = {page + 1}"
        )

        page += 1

        # ----------------------------------------------------
        # POLITE DELAY
        # ----------------------------------------------------

        time.sleep(
            random.uniform(
                1,
                2
            )
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    if os.path.exists(OUTPUT_FILE):

        final_df = pd.read_csv(
            OUTPUT_FILE
        )

        print(
            f"Total articles: "
            f"{len(final_df)}"
        )

        print(
            f"Saved: {OUTPUT_FILE}"
        )


if __name__ == "__main__":
    main()