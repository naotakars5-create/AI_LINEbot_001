"""
毎朝のAIニュース和訳ダイジェスト

流れ:
  1. feeds.yaml のRSSを集める
  2. 新着の中から「今日読む価値のある5本」をAIに選ばせる
  3. 選ばれた5本の本文を取ってきて、日本語に要約・翻訳する
  4. docs/ にHTMLページを作る（GitHub Pagesで読める）
  5. LINEに見出しだけ送る

ローカルで試すとき:
  python main.py --dry-run     … LINEに送らず、ページも書かずに標準出力へ
  python main.py --no-line     … ページは作るがLINEには送らない
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
import trafilatura
import yaml
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------- 設定

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
LOG_DIR = DOCS / "log"
STATE_PATH = ROOT / "state.json"

# 使うモデル。無料枠のあるモデルを既定にしています。
# 変えたいときは、GitHubのVariablesに GEMINI_MODEL を登録すればコード変更なしで差し替わります。
MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
PICK_COUNT = 5                        # 毎朝届く本数
MIN_JA = 2                            # うち日本語を最低何本入れるか（候補があれば）
MAX_CANDIDATES = 60                   # AIに見せる候補の上限（コスト対策）
HOURS_BACK = 36                       # 何時間前までの記事を新着とみなすか
BODY_CHARS = 6000                     # 1記事あたり読ませる本文の最大文字数
ENTRIES_PER_FEED = 30                 # 1フィードから見る記事数の上限
ARCHIVE_LINKS = 14                    # ページ下部に並べる過去日数
STATE_KEEP = 800                      # 「送信済み」を覚えておく件数
LINE_LIMIT = 4900                     # LINEの1通あたり上限（5000）の安全圏

JST = timezone(timedelta(hours=9))
UA = {"User-Agent": "Mozilla/5.0 (compatible; ai-news-bot/1.0)"}

# 返してほしいJSONの形。モデル側にこの形を守らせるので、
# 「前置きが付いていて読めない」といった崩れ方をしなくなる。
PICK_SCHEMA = {
    "type": "object",
    "properties": {"picks": {"type": "array", "items": {"type": "integer"}}},
    "required": ["picks"],
}

DIGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "articles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "title_ja": {"type": "string"},
                    "summary": {"type": "array", "items": {"type": "string"}},
                    "whats_new": {"type": "string"},
                    "terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "desc": {"type": "string"},
                            },
                            "required": ["term", "desc"],
                        },
                    },
                    "importance": {"type": "string", "enum": ["A", "B", "C"]},
                },
                "required": ["index", "title_ja", "summary", "whats_new",
                             "terms", "importance"],
            },
        }
    },
    "required": ["articles"],
}


# ---------------------------------------------------------------- 状態の保存

def load_state() -> dict:
    """前に送った記事のURLを覚えておくファイル"""
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data.get("sent"), list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [warn] state.json が読めないので作り直します: {e}", file=sys.stderr)
    return {"sent": []}


def save_state(state: dict) -> None:
    state["sent"] = state["sent"][-STATE_KEEP:]   # 増えすぎないよう古いものは捨てる
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------- 1. 収集

def collect(feeds: list, seen: set) -> list:
    """RSSを回って、まだ送っていない新着記事を集める"""
    since = datetime.now(timezone.utc) - timedelta(hours=HOURS_BACK)
    articles, urls = [], set()

    for feed in feeds:
        try:
            res = requests.get(feed["url"], headers=UA, timeout=20)
            res.raise_for_status()
            parsed = feedparser.parse(res.content)
        except Exception as e:
            print(f"  [skip] {feed['name']}: {e}", file=sys.stderr)
            continue

        count = 0
        for entry in parsed.entries[:ENTRIES_PER_FEED]:
            url = entry.get("link")
            if not url or url in seen or url in urls:
                continue

            published = entry.get("published_parsed") or entry.get("updated_parsed")
            dt = None
            if published:
                dt = datetime(*published[:6], tzinfo=timezone.utc)
                if dt < since:
                    continue

            urls.add(url)
            articles.append({
                "title": entry.get("title", "").strip(),
                "url": url,
                "source": feed["name"],
                "lang": feed.get("lang", "en"),
                "tier": feed.get("tier", "media"),
                "published": dt,
                "excerpt": strip_tags(entry.get("summary", ""))[:400],
            })
            count += 1

        print(f"  {feed['name']}: {count}本")

    return articles


def strip_tags(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def rank_candidates(articles: list) -> list:
    """AIに見せる前の並べ替え。一次情報を先に、同じ格なら新しい順。

    こうしておかないと、feeds.yaml の上のほうに書いた媒体だけが
    MAX_CANDIDATES の枠を食いつぶしてしまう。
    """
    oldest = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return sorted(
        articles,
        key=lambda a: (
            0 if a["tier"] == "primary" else 1,
            -(a["published"] or oldest).timestamp(),
        ),
    )


# ---------------------------------------------------------------- 2. 選抜

def select(client, articles: list) -> list:
    """候補の中から読む価値のある5本をAIに選ばせる"""
    if len(articles) <= PICK_COUNT:
        return articles

    candidates = rank_candidates(articles)[:MAX_CANDIDATES]
    listing = "\n".join(
        f"[{i}] ({a['lang']}/{a['tier']}) {a['source']} | {a['title']} | {a['excerpt'][:150]}"
        for i, a in enumerate(candidates)
    )

    prompt = f"""あなたはAI業界を追いかけている人のための編集者です。
以下の記事一覧から、今日読む価値のあるものを{PICK_COUNT}本だけ選んでください。

選ぶ基準:
- 各社の公式発表（tier=primary）を優先する
- 同じ話題が複数あるときは、一次情報の側を残して重複は落とす
- 単なる資金調達・株価・人事の話は優先度を下げる
- 日本語ソースを2〜3本、英語ソースを2〜3本の比率にする。
  日本語ソースは国内の動きを知るために必ず入れること

記事一覧:
{listing}

選んだ番号を picks に入れて返してください。
例: {{"picks": [3, 11, 24, 25, 40]}}"""

    try:
        raw = call_model(client, prompt, max_tokens=2000, schema=PICK_SCHEMA)
        picked = json.loads(extract_json(raw)).get("picks", [])
        chosen, used = [], set()
        for i in picked:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in used:
                used.add(i)
                chosen.append(candidates[i])
        if chosen:
            # 5本に満たなければ候補の先頭から補う
            for a in candidates:
                if len(chosen) >= PICK_COUNT:
                    break
                if a not in chosen:
                    chosen.append(a)
            return ensure_japanese(chosen[:PICK_COUNT], candidates)
        raise ValueError("有効な番号が返ってきませんでした")
    except Exception as e:
        print(f"  [warn] 選抜に失敗したので先頭から取ります: {e}", file=sys.stderr)
        return ensure_japanese(candidates[:PICK_COUNT], candidates)


def ensure_japanese(chosen: list, candidates: list) -> list:
    """日本語の記事が少なすぎたら、英語の下位と入れ替えて必ず入れる。

    プロンプトで比率を頼んでも守られないことがあるので、最後にコードで担保する。
    候補に日本語がそもそも無い日は、無いなりの本数で通す。
    """
    have = sum(1 for a in chosen if a["lang"] == "ja")
    if have >= MIN_JA:
        return chosen

    urls = {a["url"] for a in chosen}
    spare = [a for a in candidates if a["lang"] == "ja" and a["url"] not in urls]
    out = list(chosen)
    while have < MIN_JA and spare:
        # 入れ替える相手は、英語のうち一番うしろにあるもの
        victim = next((i for i in range(len(out) - 1, -1, -1)
                       if out[i]["lang"] != "ja"), None)
        if victim is None:
            break
        swapped = spare.pop(0)
        print(f"  [調整] 日本語を1本追加: {swapped['title'][:40]}", file=sys.stderr)
        out[victim] = swapped
        have += 1
    return out


# ---------------------------------------------------------------- 3. 本文取得と和訳

def fetch_body(url: str) -> str:
    """記事の本文だけを抜き出す。取れなければ空文字"""
    try:
        res = requests.get(url, headers=UA, timeout=20)
        res.raise_for_status()
        body = trafilatura.extract(
            res.text, include_comments=False, include_tables=False
        )
        return (body or "")[:BODY_CHARS]
    except Exception as e:
        print(f"  [warn] 本文が取れません {url}: {e}", file=sys.stderr)
        return ""


def summarize(client, articles: list) -> list:
    """5本まとめて1回のAPI呼び出しで日本語化する"""
    blocks = []
    for i, a in enumerate(articles):
        body = fetch_body(a["url"])
        a["has_body"] = bool(body)
        blocks.append(
            f"=== 記事{i} ===\n"
            f"媒体: {a['source']}\n"
            f"原題: {a['title']}\n"
            f"本文: {body or a['excerpt'] or '(本文が取得できませんでした)'}"
        )

    prompt = f"""以下の記事を日本語にしてください。英語の記事は翻訳しながら要約します。

{chr(10).join(blocks)}

各記事について、articles に次の形で入れて返してください。

{{
  "articles": [
    {{
      "index": 0,
      "title_ja": "日本語に訳したタイトル",
      "summary": ["要約1行目", "要約2行目", "要約3行目"],
      "whats_new": "これまでと比べて何が新しくなったのかを1〜2行で",
      "terms": [{{"term": "専門用語", "desc": "1行の説明"}}],
      "importance": "A"
    }}
  ]
}}

書き方のルール:
- summary は3行ちょうど。1行は40〜60字程度
- whats_new は「内容の紹介」ではなく「前との差分」を書く。差分が読み取れないなら記事の位置づけを書く
- terms はその記事に出てきた専門用語を2〜3個。初めて見る人向けの説明にする
- 本文が取得できていない記事は、タイトルと抜粋から分かる範囲で書き、無理に断定しない

importance は次の基準で厳密に付けてください。印象ではなく、条件に当てはまるかで決めます。
- A … 新しいモデル・製品・機能が「いま実際に使えるようになった」発表。
      読んだ人がその日のうちに試したり、仕事のやり方を変えたりしうるもの。
- B … 重要だが、いますぐ動く必要はないもの。
      研究成果、方針や計画の表明、提携・買収、規制やルールの動き。
- C … 上記以外。解説・考察・まとめ・体験記など、背景を知るための記事。
迷ったら低いほうを付けてください。Aは1日に0〜2本程度が適正です。"""

    try:
        raw = call_model(client, prompt, max_tokens=16000, schema=DIGEST_SCHEMA)
        data = json.loads(extract_json(raw)).get("articles", [])
    except Exception as e:
        # 和訳に失敗しても、原題だけでもLINEに届けたい
        print(f"  [error] 和訳に失敗しました。原題のまま出します: {e}", file=sys.stderr)
        return [{**a, "title_ja": a["title"], "summary": [], "whats_new": "",
                 "terms": [], "importance": "B"} for a in articles]

    items, used = [], set()
    for entry in data if isinstance(data, list) else []:
        i = entry.get("index")
        if isinstance(i, int) and 0 <= i < len(articles) and i not in used:
            used.add(i)
            items.append({**articles[i], **entry})

    # AIが取りこぼした記事も落とさずに並べる
    for i, a in enumerate(articles):
        if i not in used:
            items.append({**a, "title_ja": a["title"], "summary": [],
                          "whats_new": "", "terms": [], "importance": "B"})
    return items


RETRYABLE = {429, 500, 502, 503, 504}   # 混雑・一時的な障害。待てば直る


def call_model(client, prompt: str, max_tokens: int, schema: dict,
               attempts: int = 3) -> str:
    """一時的な失敗は少し待ってやり直す。設定ミスなど直らない失敗はすぐ諦める"""
    for n in range(attempts):
        try:
            res = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    # 道具は渡していないので切っておく（警告がログに出るのを防ぐ）
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
        except genai_errors.APIError as e:
            if e.code == 404:
                raise SystemExit(explain_missing_model(client, e)) from e
            if e.code not in RETRYABLE or n == attempts - 1:
                raise
            wait = 2 ** (n + 1)
            print(f"  [retry] {e.code} {e.status}: {wait}秒待ちます", file=sys.stderr)
            time.sleep(wait)
            continue

        text = (res.text or "").strip()
        if text:
            return text
        # 出力枠を使い切って本文が空、というのが一番ありがちな失敗
        raise ValueError(f"応答が空です（finish_reason={finish_reason_of(res)}）。"
                         f"max_output_tokens を増やすと直ることがあります")
    raise RuntimeError("到達しません")


def finish_reason_of(res) -> str:
    try:
        return str(res.candidates[0].finish_reason)
    except (AttributeError, IndexError, TypeError):
        return "不明"


def explain_missing_model(client, err) -> str:
    """モデル名が違うときに、使える名前を並べて教える"""
    lines = [f"[error] モデル '{MODEL}' が見つかりません（{err.code} {err.status}）。"]
    try:
        names = sorted(
            m.name.removeprefix("models/") for m in client.models.list()
            if "generateContent" in (m.supported_actions or [])
        )
        if names:
            lines.append("いま使えるモデル: " + ", ".join(names))
    except Exception:
        pass
    lines.append("GitHubのVariablesに GEMINI_MODEL を登録すると差し替えられます。")
    return "\n".join(lines)


def extract_json(raw: str) -> str:
    """```json ``` で囲まれていても中身だけ取り出す"""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    start = min([p for p in (raw.find("["), raw.find("{")) if p >= 0], default=0)
    end = max(raw.rfind("]"), raw.rfind("}"))
    return raw[start:end + 1]


# ---------------------------------------------------------------- 4. ページ生成

def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render(items: list, date: datetime, archive: list, depth: int) -> str:
    """1日分のHTMLを組み立てる。depth は docs/ までの相対階層"""
    base = "../" * depth
    weekday = "月火水木金土日"[date.weekday()]
    articles_html = []

    for n, it in enumerate(items, 1):
        terms = "".join(
            f'<div class="term"><span class="term-name">{esc(t.get("term",""))}</span>'
            f'<span class="term-desc">{esc(t.get("desc",""))}</span></div>'
            for t in (it.get("terms") or []) if isinstance(t, dict)
        )
        summary = "".join(f"<p>{esc(line)}</p>" for line in (it.get("summary") or []))
        articles_html.append(f"""
      <article class="entry" id="n{n}" style="--n:{n}">
        <div class="entry-meta">
          <span class="rank" data-level="{esc(it.get('importance','B'))}">{esc(it.get('importance','B'))}</span>
          <span class="source">{esc(it['source'])}</span>
        </div>
        <p class="original">{esc(it['title'])}</p>
        <h2 class="translated">{esc(it.get('title_ja') or it['title'])}</h2>
        <div class="summary">{summary}</div>
        {f'''<div class="diff">
          <span class="diff-label">何が変わったか</span>
          <p>{esc(it.get("whats_new",""))}</p>
        </div>''' if it.get("whats_new") else ''}
        {f'<div class="terms">{terms}</div>' if terms else ''}
        <a class="link" href="{esc(it['url'])}" target="_blank" rel="noopener">原文を読む</a>
      </article>""")

    archive_html = "".join(
        f'<a href="{base}log/{d}.html">{d[5:].replace("-", "/")}</a>'
        for d in archive[:ARCHIVE_LINKS]
    )

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Digest {date:%Y-%m-%d}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Shippori+Mincho+B1:wght@500;700&family=Zen+Kaku+Gothic+New:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{base}style.css">
</head>
<body>
  <div class="rail"><span class="rail-date">{date:%Y.%m.%d}</span></div>
  <main>
    <header class="head">
      <p class="eyebrow">AI DIGEST</p>
      <h1>{date.month}月{date.day}日<span class="wd">（{weekday}）</span></h1>
      <p class="lede">今朝の{len(items)}本。海外の一次情報と国内の動きを、日本語に直して並べています。</p>
      <p class="legend">
        <span class="rank" data-level="A">A</span>すぐ使える
        <span class="rank" data-level="B">B</span>知っておく
        <span class="rank" data-level="C">C</span>参考
      </p>
    </header>
    {"".join(articles_html)}
    <footer class="foot">
      <p class="foot-label">過去の分</p>
      <nav class="archive">{archive_html}</nav>
    </footer>
  </main>
</body>
</html>"""


def build_pages(items: list, now: datetime) -> str:
    """今日のページと index を書き出す。戻り値は公開URL用の日付文字列"""
    today = now.strftime("%Y-%m-%d")
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    dates = sorted(
        {p.stem for p in LOG_DIR.glob("*.html")} | {today}, reverse=True
    )

    # log/ の中は docs/ からひとつ下、index.html は docs/ 直下
    (LOG_DIR / f"{today}.html").write_text(
        render(items, now, dates, depth=1), encoding="utf-8"
    )
    (DOCS / "index.html").write_text(
        render(items, now, dates, depth=0), encoding="utf-8"
    )
    return today


# ---------------------------------------------------------------- 5. LINE送信

RANK_COLOR = {"A": "#C98A12", "B": "#2F4858", "C": "#6B7770"}
RANK_LABEL = {"A": "すぐ使える", "B": "知っておく", "C": "参考"}


def is_http_url(u) -> bool:
    return isinstance(u, str) and u.startswith(("http://", "https://"))


def button(label: str, uri: str, primary: bool = False) -> dict:
    b = {"type": "button", "height": "sm",
         "action": {"type": "uri", "label": label[:20], "uri": uri}}
    if primary:
        b["style"], b["color"] = "primary", "#2E6F5E"
    else:
        b["style"] = "link"
    return b


def build_flex(items: list, now: datetime, page_url: str) -> dict:
    """1記事1枚のカードを横に並べる（LINEのFlex Message / carousel）"""
    weekday = "月火水木金土日"[now.weekday()]
    bubbles = []

    for n, it in enumerate(items, 1):
        rank = str(it.get("importance", "B")).upper()
        if rank not in RANK_COLOR:
            rank = "B"
        title = (it.get("title_ja") or it.get("title") or "").strip() or "(無題)"
        head = ((it.get("summary") or [""])[0] or "").strip()

        body = [{"type": "text", "text": title[:120], "weight": "bold",
                 "size": "md", "wrap": True, "maxLines": 4, "color": "#22272A"}]
        if head:
            body.append({"type": "text", "text": head[:200], "size": "sm",
                         "wrap": True, "maxLines": 4, "margin": "md",
                         "color": "#5A6469"})

        footer = []
        if is_http_url(it.get("url")):
            footer.append(button("原文", it["url"]))
        if is_http_url(page_url):
            footer.append(button("くわしく", f"{page_url}#n{n}"))

        bubble = {
            "type": "bubble",
            "size": "kilo",
            "header": {
                "type": "box", "layout": "horizontal", "spacing": "sm",
                "paddingAll": "10px", "backgroundColor": RANK_COLOR[rank],
                "contents": [
                    {"type": "text", "text": f"{rank}・{RANK_LABEL[rank]}",
                     "size": "xs", "weight": "bold", "color": "#FFFFFF", "flex": 0},
                    {"type": "text", "text": str(it.get("source", ""))[:22],
                     "size": "xs", "color": "#FFFFFF", "align": "end",
                     "gravity": "center"},
                ],
            },
            "body": {"type": "box", "layout": "vertical", "paddingAll": "14px",
                     "contents": body},
        }
        if footer:
            bubble["footer"] = {"type": "box", "layout": "horizontal",
                                "spacing": "sm", "paddingAll": "6px",
                                "contents": footer}
        bubbles.append(bubble)

    # 最後に、全文ページへ渡すための1枚
    if is_http_url(page_url):
        bubbles.append({
            "type": "bubble", "size": "kilo",
            "body": {
                "type": "box", "layout": "vertical", "spacing": "md",
                "paddingAll": "18px",
                "contents": [
                    {"type": "text", "text": f"{now.month}月{now.day}日（{weekday}）",
                     "size": "sm", "color": "#5A6469"},
                    {"type": "text", "text": f"今朝の{len(items)}本", "size": "xl",
                     "weight": "bold", "color": "#22272A"},
                    {"type": "text",
                     "text": "3行要約・何が変わったか・専門用語の解説つき",
                     "size": "sm", "wrap": True, "color": "#5A6469"},
                    {"type": "separator", "margin": "md"},
                    {"type": "text",
                     "text": "A＝すぐ使える  B＝知っておく  C＝参考",
                     "size": "xs", "wrap": True, "color": "#8A9296"},
                ],
            },
            "footer": {"type": "box", "layout": "vertical", "paddingAll": "6px",
                       "contents": [button("全文の和訳を読む", page_url, primary=True)]},
        })

    return {"type": "flex", "altText": build_alt(items, now),
            "contents": {"type": "carousel", "contents": bubbles}}


def build_alt(items: list, now: datetime) -> str:
    """通知欄に出る文字。400字までなので短く"""
    weekday = "月火水木金土日"[now.weekday()]
    lines = [f"AI Digest {now.month}/{now.day}（{weekday}）"]
    for n, it in enumerate(items, 1):
        lines.append(f"{n}. {it.get('title_ja') or it.get('title','')}")
    text = "\n".join(lines)
    return text[:390] + "…" if len(text) > 390 else text


def push_line(messages: list) -> bool:
    token = os.environ["LINE_CHANNEL_ACCESS_TOKEN"]
    user_id = os.environ["LINE_USER_ID"]
    try:
        res = requests.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            json={"to": user_id, "messages": messages},
            timeout=20,
        )
    except requests.RequestException as e:
        print(f"  [error] LINEに繋がりません: {e}", file=sys.stderr)
        return False
    if res.status_code != 200:
        print(f"  [error] LINE送信に失敗: {res.status_code} {res.text}",
              file=sys.stderr)
        return False
    return True


def send_digest(items: list, now: datetime, page_url: str) -> None:
    """カードで送る。カードが弾かれたら文字だけで送り直す"""
    if push_line([build_flex(items, now, page_url)]):
        print("  LINEにカードで送信しました")
        return
    print("  [warn] カードで送れなかったので、文字だけで送り直します", file=sys.stderr)
    if push_line([{"type": "text", "text": build_message(items, now, page_url)}]):
        print("  LINEに送信しました（文字のみ）")
        return
    sys.exit(1)


def build_message(items: list, now: datetime, page_url: str) -> str:
    """上限を超えそうなら、要約の1行目から削って必ずリンクを残す"""
    weekday = "月火水木金土日"[now.weekday()]
    header = f"AI Digest {now.month}/{now.day}（{weekday}）\n\n"
    footer = f"▼全文の和訳はこちら\n{page_url}"

    def compose(with_summary: bool) -> str:
        lines = []
        for n, it in enumerate(items, 1):
            lines.append(f"{n}. [{it.get('importance','B')}] "
                         f"{it.get('title_ja') or it['title']}")
            head = (it.get("summary") or [""])[0]
            if with_summary and head:
                lines.append(f"   {head}")
            lines.append("")
        return header + "\n".join(lines) + footer

    full = compose(True)
    if len(full) <= LINE_LIMIT:
        return full
    short = compose(False)
    return short if len(short) <= LINE_LIMIT else short[:LINE_LIMIT]


# ---------------------------------------------------------------- 実行

def require_env(names: list) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"[error] 環境変数が足りません: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="ページを書かず、LINEにも送らず、結果を画面に出すだけ")
    ap.add_argument("--no-line", action="store_true", help="LINE送信だけしない")
    args = ap.parse_args()

    send_line = not (args.dry_run or args.no_line)
    require_env(["GEMINI_API_KEY"]
                + (["LINE_CHANNEL_ACCESS_TOKEN", "LINE_USER_ID"] if send_line else []))

    now = datetime.now(JST)
    print(f"=== AI Digest {now:%Y-%m-%d %H:%M} ===")

    feeds = yaml.safe_load((ROOT / "feeds.yaml").read_text(encoding="utf-8"))["feeds"]
    state = load_state()
    seen = set(state["sent"])

    print("[1] 収集")
    articles = collect(feeds, seen)
    print(f"  → 新着 {len(articles)}本")

    if not articles:
        print("新着がないので終了します")
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    print(f"  モデル: {MODEL}")

    print("[2] 選抜")
    picked = select(client, articles)
    for a in picked:
        print(f"  - {a['title'][:60]}")

    print("[3] 和訳")
    items = summarize(client, picked)

    if args.dry_run:
        print("[4] ページ生成（--dry-run のため書き出しません）")
        for n, it in enumerate(items, 1):
            print(f"\n{n}. [{it.get('importance','B')}] "
                  f"{it.get('title_ja') or it['title']}")
            for line in it.get("summary") or []:
                print(f"   {line}")
            print(f"   {it['url']}")
        return

    print("[4] ページ生成")
    date_str = build_pages(items, now)

    base_url = os.environ.get("PAGES_URL", "").rstrip("/")
    page_url = f"{base_url}/log/{date_str}.html" if base_url else "(PAGES_URLが未設定です)"

    if send_line:
        print("[5] LINE送信")
        send_digest(items, now, page_url)
    else:
        print("[5] LINE送信（--no-line のため送りません）")
        print(build_message(items, now, page_url))

    state["sent"].extend(a["url"] for a in picked)
    save_state(state)
    print("完了")


if __name__ == "__main__":
    main()
