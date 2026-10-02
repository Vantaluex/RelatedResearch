"""
Connected Papers CLI & Dual Literature Graph Engine
===================================================
A standalone research tool for generating interactive dual citation graphs
(Bleeding Edge contemporaries & Foundational Roots ancestors) using the Semantic Scholar API.

Security & Environment Note:
- Never hardcode API keys in this script.
- Store your key in a `.env` file as:
    S2_API_KEY=your_actual_key_here
- Ensure `.env` is listed in your `.gitignore` before committing to Git!
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import webbrowser

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Attempt to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    # Load .env file from script directory or workspace root
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# =============================================================================
# CONFIGURATION & CLIENT
# =============================================================================
# Read API key securely from environment variable
API_KEY = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or ""

BASE_URL = "https://api.semanticscholar.org/graph/v1"
REC_URL = "https://api.semanticscholar.org/recommendations/v1"

CACHE_DIR = Path(".cache_s2")
CACHE_DIR.mkdir(exist_ok=True)


class S2Client:
    def __init__(self, api_key: str = "", min_delay: float = None):
        self.api_key = (api_key or "").strip()
        # Authenticated requests can use ~1.0-1.8s delay; unauthenticated needs ~3.0s to avoid 429
        if min_delay is None:
            self.min_delay = 1.2 if self.api_key else 3.0
        else:
            self.min_delay = min_delay

        self.last_response_time = 0.0
        self.session = requests.Session()
        
        headers = {
            "User-Agent": "ConnectedPapersTool/28.0 (Academic Research; Clean-Env-Build)"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        self.session.headers.update(headers)

    def _get_cache_path(self, method: str, url: str, params: dict, json_data: dict) -> Path:
        # Deterministic serialization for hashing avoiding unhashable dict/list comparison bugs
        p_str = json.dumps(params, sort_keys=True, default=str) if params else ""
        j_str = json.dumps(json_data, sort_keys=True, default=str) if json_data else ""
        raw = f"{method}:{url}:{p_str}:{j_str}"
        key = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{key}.json"

    def _throttle(self):
        elapsed = time.time() - self.last_response_time
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

    def request(self, method: str, url: str, max_retries: int = 4, **kwargs):
        params = kwargs.get("params")
        json_data = kwargs.get("json")
        cache_file = self._get_cache_path(method, url, params, json_data)

        # 1. Check local cache (stored in human-readable JSON instead of arbitrary pickle execution)
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        full_url = f"{BASE_URL}{url}" if url.startswith("/") else url
        backoff = 3.0

        for attempt in range(max_retries + 1):
            self._throttle()
            try:
                res = self.session.request(method, full_url, timeout=25, **kwargs)
            except requests.RequestException as e:
                if attempt == max_retries:
                    print(f"[!] Network error on {url}: {e}")
                    return None
                print(f"[!] Connection failed. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= 2.0
                continue
            finally:
                self.last_response_time = time.time()

            if res.status_code == 200:
                try:
                    data = res.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    return data
                except Exception as e:
                    print(f"[!] Failed to parse or cache JSON: {e}")
                    return None

            if res.status_code == 429:
                retry_header = res.headers.get("Retry-After")
                wait_time = float(retry_header) if retry_header else backoff
                print(f"[!] 429 Throttled by Semantic Scholar. Cooling down {wait_time:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                backoff = max(wait_time * 2.0, backoff * 2.0)
                continue

            if res.status_code in (401, 403):
                if self.api_key:
                    print(f"[!] HTTP {res.status_code}: API key was rejected or unauthorized. Please verify S2_API_KEY in .env.")
                else:
                    print(f"[!] HTTP {res.status_code}: Request unauthorized. Consider adding S2_API_KEY to your .env file.")
                return None

            print(f"[!] HTTP {res.status_code} on {url}: {res.text[:120]}")
            return None

        print(f"[!] Exhausted {max_retries} attempts on {url}.")
        return None

    def get(self, url: str, params: dict = None):
        return self.request("GET", url, params=params)

    def post(self, url: str, json_data: dict, params: dict = None):
        return self.request("POST", url, json=json_data, params=params)


client = S2Client(API_KEY)


# =============================================================================
# CONFERENCE & PREPRINT TIER CLASSIFICATION ENGINE
# =============================================================================
def classify_conference_venue(venue_str: str, pdf_url: str = "", paper_url: str = ""):
    v = (venue_str or "").strip()
    v_lower = v.lower()
    urls = f"{pdf_url} {paper_url}".lower()

    # Preprints detection: ONLY true unreviewed preprints / working papers
    preprint_patterns = [
        r'\barxiv\b', r'\bcorr\b', r'\bbiorxiv\b', r'\bmedrxiv\b', r'\bchemrxiv\b',
        r'\bssrn\b', r'\btech(nical)?\s*report\b', r'\bpreprint\b', r'\bworking\s*paper\b',
        r'\bresearch\s*square\b', r'\bpreprints\.org\b'
    ]
    is_preprint = any(re.search(pat, v_lower) for pat in preprint_patterns) or \
                  any(re.search(pat, urls) for pat in [r'arxiv\.org', r'biorxiv\.org', r'medrxiv\.org', r'ssrn\.com'])

    # 1. RED: Top-Tier Architecture Big 4
    for abbr in ["asplos", "hpca", "isca"]:
        if re.search(rf"\b{abbr}\b", v_lower):
            return {"tier": "top_arch", "color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "abbr": abbr.upper(), "tier_name": "Top Arch"}
    if re.search(r"\b(microarchitecture|micro)\b", v_lower) and not re.search(r"\b(ieee micro|microbiology|microscopy)\b", v_lower):
        return {"tier": "top_arch", "color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "abbr": "MICRO", "tier_name": "Top Arch"}
    if "computer architecture" in v_lower and ("symposium" in v_lower or "international" in v_lower):
        if "high-performance" in v_lower or "high performance" in v_lower:
            return {"tier": "top_arch", "color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "abbr": "HPCA", "tier_name": "Top Arch"}
        elif "architectural support" in v_lower or "programming languages" in v_lower:
            return {"tier": "top_arch", "color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "abbr": "ASPLOS", "tier_name": "Top Arch"}
        else:
            return {"tier": "top_arch", "color": "#ef4444", "bg": "#fef2f2", "border": "#fca5a5", "abbr": "ISCA", "tier_name": "Top Arch"}

    # 2. PURPLE: Top Systems & Core CS (Score = 1.00)
    purple_checks = [
        (r"\b(sosp|symposium on operating systems principles)\b", "SOSP"),
        (r"\b(osdi|operating systems design and implementation)\b", "OSDI"),
        (r"\b(nsdi|networked systems design)\b", "NSDI"),
        (r"\b(sigcomm)\b", "SIGCOMM"),
        (r"\b(sigmod)\b", "SIGMOD"),
        (r"\b(vldb|pldb|pvlldb)\b", "VLDB"),
        (r"\b(pldi|programming language design and implementation)\b", "PLDI"),
        (r"\b(popl|principles of programming languages)\b", "POPL"),
        (r"\b(icse|international conference on software engineering)\b", "ICSE"),
        (r"\b(fse|esec/fse)\b", "FSE"),
        (r"\b(oopsla)\b", "OOPSLA"),
        (r"\b(ieee s&p|oakland|symposium on security and privacy)\b", "S&P"),
        (r"\b(usenix sec|usenix security)\b", "USENIX Sec"),
        (r"\b(acm ccs|computer and communications security)\b", "CCS"),
        (r"\b(neurips|nips)\b", "NeurIPS"),
        (r"\b(icml|international conference on machine learning)\b", "ICML"),
        (r"\b(iclr|learning representations)\b", "ICLR"),
        (r"\b(cvpr)\b", "CVPR"),
        (r"\b(iccv)\b", "ICCV"),
        (r"\b(kdd)\b", "KDD"),
        (r"\b(stoc)\b", "STOC"),
        (r"\b(focs)\b", "FOCS"),
        (r"\b(siggraph)\b", "SIGGRAPH"),
        (r"\b(chi)\b", "CHI"),
        (r"\b(aaai)\b", "AAAI"),
        (r"\b(acl)\b", "ACL")
    ]
    for pattern, name in purple_checks:
        if re.search(pattern, v_lower):
            return {"tier": "top_other", "color": "#a855f7", "bg": "#faf5ff", "border": "#d8b4fe", "abbr": name, "tier_name": "Top CS/Systems"}

    # 3. ORANGE: Premier Arch / Systems / EDA (Score 0.70 - 0.99)
    orange_checks = [
        (r"\b(dac|design automation conference)\b", "DAC"),
        (r"\b(date|design,?\s+automation\s+and\s+test)\b", "DATE"),
        (r"\b(iccad|computer-aided design|computer aided design)\b", "ICCAD"),
        (r"\b(asp-dac|aspdac)\b", "ASP-DAC"),
        (r"\b(eurosys|european conference on computer systems)\b", "EuroSys"),
        (r"\b(usenix atc|atc|annual technical conference)\b", "USENIX ATC"),
        (r"\b(fast|file and storage technologies)\b", "FAST"),
        (r"\b(ppopp|principles and practice of parallel programming)\b", "PPoPP"),
        (r"\b(supercomputing|\bsc\s*\'?\d{0,2}\b|high performance computing, networking, storage)\b", "SC"),
        (r"\b(pact|parallel architectures and compilation)\b", "PACT"),
        (r"\b(sigmetrics|measurement and modeling of computer systems)\b", "SIGMETRICS"),
        (r"\b(cgo|code generation and optimization)\b", "CGO"),
        (r"\b(iccd|computer design)\b", "ICCD"),
        (r"\b(iiswc|workload characterization)\b", "IISWC"),
        (r"\b(ispass|performance analysis of systems)\b", "ISPASS"),
        (r"\b(rtss|real-time systems symposium)\b", "RTSS"),
        (r"\b(rtas|real-time and embedded)\b", "RTAS"),
        (r"\b(fccm|field-programmable custom)\b", "FCCM"),
        (r"\b(fpga)\b", "FPGA"),
        (r"\b(mobisys)\b", "MobiSys"),
        (r"\b(mobicom)\b", "MobiCom"),
        (r"\b(infocom)\b", "INFOCOM"),
        (r"\b(conext)\b", "CoNEXT"),
        (r"\b(socc|symposium on cloud computing)\b", "SoCC"),
        (r"\b(hotstorage)\b", "HotStorage"),
        (r"\b(hotnets)\b", "HotNets"),
        (r"\b(ipdps)\b", "IPDPS")
    ]
    for pattern, name in orange_checks:
        if re.search(pattern, v_lower):
            if name == "SC" and not re.search(r"\b(supercomputing|high performance computing|sc\s*\'?\d+)\b", v_lower):
                continue
            return {"tier": "regarded_arch", "color": "#f97316", "bg": "#fff7ed", "border": "#fdba74", "abbr": name, "tier_name": "Premier Arch/Systems"}

    # 4. GRAY: True Preprints ONLY (Unreviewed / In-flight papers)
    if is_preprint:
        return {
            "tier": "preprint",
            "color": "#94a3b8",
            "bg": "#f8fafc",
            "border": "#cbd5e1",
            "abbr": "Preprint",
            "tier_name": "Preprint (Unreviewed)"
        }

    # 5. NO BADGE: Peer-reviewed into other / lower-tier venues
    return None


def normalize_title(title: str) -> str:
    """Canonical representation of a title to eliminate duplicate database IDs."""
    return re.sub(r'[^a-z0-9]', '', (title or "").lower())


def format_author_year(paper: dict) -> str:
    year = paper.get("year") or 2024
    authors = paper.get("authors") or []
    if authors:
        first = authors[0]
        name = first if isinstance(first, str) else first.get("name", "")
        if name:
            last_name = name.strip().split()[-1]
            return f"{last_name}, {year}"
    title = paper.get("title") or "Unknown"
    return f"{title[:12]}..., {year}"


def resolve_paper(query: str) -> str:
    """
    Resolves a title, DOI, or arXiv link to a Semantic Scholar paper ID.
    Enhanced with URL parsing for doi.org and arxiv.org web addresses.
    """
    clean = query.strip()
    
    # 1. Parse arXiv URLs and prefixes: https://arxiv.org/abs/2301.12345
    arxiv_url_match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+(?:v[0-9]+)?)', clean, re.IGNORECASE)
    if arxiv_url_match:
        return f"ARXIV:{arxiv_url_match.group(1)}"
    if clean.lower().startswith("arxiv:"):
        return f"ARXIV:{clean.split(':', 1)[1].strip()}"

    # 2. Parse DOI URLs: https://doi.org/10.1145/...
    doi_match = re.search(r'(10\.[0-9]{4,9}/[-._;()/:A-Za-z0-9]+)', clean)
    if doi_match:
        return f"DOI:{doi_match.group(1)}"

    # 3. Direct 40-character SHA paperId
    if len(clean) == 40 and all(c in "0123456789abcdefABCDEF" for c in clean):
        return clean

    # 4. Search query by title / keywords
    sanitized = re.sub(r'[:\-+!(){}\[\]^"~*?\\/]', ' ', clean)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    print(f"[*] Resolving seed paper: '{sanitized[:60]}...'")
    res = client.get("/paper/search", params={
        "query": sanitized,
        "limit": 1,
        "fields": "paperId,title,year,authors,citationCount,venue,publicationVenue"
    })

    if res and res.get("data"):
        hit = res["data"][0]
        print(f"[+] Match found: '{hit.get('title')}' ({hit.get('year')})")
        return hit["paperId"]
    return None


def extract_venue(paper: dict) -> str:
    pub_venue = paper.get("publicationVenue")
    if pub_venue and isinstance(pub_venue, dict) and pub_venue.get("name"):
        return str(pub_venue["name"]).strip()
    if paper.get("venue"):
        return str(paper["venue"]).strip()
    journal = paper.get("journal")
    if journal and isinstance(journal, dict) and journal.get("name"):
        return str(journal["name"]).strip()
    return ""


def extract_clean_domain_query(title: str) -> str:
    clean_title = re.sub(r'^[A-Za-z0-9\-_]+\s*:\s*', '', title or "")
    words = re.sub(r'[^a-zA-Z0-9\s-]', ' ', clean_title.lower()).split()
    stopwords = {
        "a", "an", "the", "and", "or", "for", "with", "in", "on", "at", "by", "to", "from",
        "towards", "enhancing", "improving", "efficient", "novel", "framework", "system", "design"
    }
    meaningful = [w for w in words if w not in stopwords and len(w) > 2]
    return " ".join(meaningful[:5])


def interpolate_connected_papers_palette(t: float) -> str:
    t = max(0.0, min(1.0, t))
    stops = [
        (0.00, (212, 224, 224)),
        (0.50, (107, 153, 158)),
        (1.00, (38, 75, 82))
    ]
    for i in range(len(stops) - 1):
        t1, c1 = stops[i]
        t2, c2 = stops[i + 1]
        if t1 <= t <= t2:
            ratio = (t - t1) / max(1e-5, (t2 - t1))
            r = int(c1[0] + ratio * (c2[0] - c1[0]))
            g = int(c1[1] + ratio * (c2[1] - c1[1]))
            b = int(c1[2] + ratio * (c2[2] - c1[2]))
            return f"rgb({r},{g},{b})"
    return "rgb(38,75,82)"


# =============================================================================
# TOPOLOGICAL SUBGRAPH PIPELINE (ZERO DUPLICATES GUARANTEED)
# =============================================================================
def process_subgraph(papers, seed_id, seed_refs_ids, seed_cites_ids):
    # Strict title-level deduplication
    unique_papers = []
    seen_titles = set()

    for p in papers:
        norm_t = normalize_title(p.get("title", ""))
        if not norm_t:
            continue
        if norm_t in seen_titles:
            continue
        seen_titles.add(norm_t)
        unique_papers.append(p)

    n = len(unique_papers)
    if n == 0:
        return [], [], [], 2020, 2026

    parsed = []
    for p in unique_papers:
        pid = p["paperId"]
        is_seed = (pid == seed_id)
        
        # Safely normalize authors list
        raw_authors = p.get("authors") or []
        authors = []
        for a in raw_authors:
            if isinstance(a, dict) and a.get("name"):
                authors.append(a["name"])
            elif isinstance(a, str):
                authors.append(a)

        # Safely retrieve PDF URL
        pdf_url = ""
        if isinstance(p.get("openAccessPdf"), dict):
            pdf_url = p.get("openAccessPdf", {}).get("url") or ""
        elif p.get("pdf_url"):
            pdf_url = str(p.get("pdf_url"))

        p_refs = {r["paperId"] for r in (p.get("references") or []) if isinstance(r, dict) and r.get("paperId")}
        venue_str = extract_venue(p)
        conf_badge = classify_conference_venue(venue_str, pdf_url=pdf_url, paper_url=p.get("url") or "")

        # Safe int conversion for citation count and year (guards against None from API)
        raw_cites = p.get("citationCount")
        if raw_cites is None:
            raw_cites = p.get("citations")
        citation_count = int(raw_cites) if (raw_cites is not None and str(raw_cites).isdigit()) else 0

        year_val = p.get("year")
        year = int(year_val) if (year_val is not None and str(year_val).isdigit()) else 2024

        parsed.append({
            "id": pid,
            "title": p.get("title") or "Untitled",
            "year": year,
            "citations": citation_count,
            "venue": venue_str,
            "conf_badge": conf_badge,
            "abstract": p.get("abstract") or "No abstract indexed in Semantic Scholar.",
            "authors": authors,
            "label": format_author_year({"year": year, "authors": authors, "title": p.get("title")}),
            "pdf_url": pdf_url,
            "refs": p_refs,
            "is_seed": is_seed
        })

    # TF-IDF Cosine Similarity Matrix with Empty Vocabulary Protection
    corpus = [f"{p['title']} {p['abstract']} {' '.join(p['authors'])}" for p in parsed]
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
        text_sim = cosine_similarity(tfidf)
    except ValueError:
        # Fallback if corpus vocabulary is completely empty or filtered out by stopwords
        text_sim = np.eye(n)

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim_matrix[i][j] = 1.0
                continue
            sim = float(text_sim[i, j])
            
            # Boost similarity between seed and direct references / citations
            u_is_seed = parsed[i]["is_seed"]
            v_is_seed = parsed[j]["is_seed"]

            if (u_is_seed and parsed[j]["id"] in seed_refs_ids) or (v_is_seed and parsed[i]["id"] in seed_refs_ids):
                sim = max(sim, 0.44)
            if (u_is_seed and parsed[j]["id"] in seed_cites_ids) or (v_is_seed and parsed[i]["id"] in seed_cites_ids):
                sim = max(sim, 0.44)

            t_i, t_j = parsed[i]["title"].lower(), parsed[j]["title"].lower()
            overlap = len(set(t_i.split()) & set(t_j.split()))
            if overlap >= 2:
                sim = min(1.0, sim + 0.14)

            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Percentile Normalization with Safe Divisor
    off_diag = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    p_low = np.percentile(off_diag, 10) if len(off_diag) > 0 else 0.0
    p_high = np.percentile(off_diag, 90) if len(off_diag) > 0 else 1.0
    denom = max(1e-4, p_high - p_low)

    stretched_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                stretched_sim[i][j] = 1.0
            else:
                norm_val = np.clip((sim_matrix[i][j] - p_low) / denom, 0.0, 1.0)
                stretched_sim[i][j] = float(norm_val ** 2.0)

    years = [p["year"] for p in parsed if isinstance(p["year"], int)]
    min_y, max_y = (min(years), max(years)) if years else (2020, 2026)
    citations_list = [p["citations"] for p in parsed]
    min_c, max_c = min(citations_list), max(citations_list)
    cit_spread = max(1e-4, (math.sqrt(max_c) - math.sqrt(min_c)))

    nodes_payload = []
    meta_payload = []

    for p in parsed:
        norm_cit = (math.sqrt(p["citations"]) - math.sqrt(min_c)) / cit_spread
        p["size"] = int(24 + norm_cit * 28)

        norm_y = (p["year"] - min_y) / max(1, (max_y - min_y))
        color = interpolate_connected_papers_palette(norm_y)

        if p["is_seed"]:
            p["size"] = max(p["size"], 42)
            node_color = {
                "background": "#7e22ce",
                "border": "#e879f9",
                "highlight": {"background": "#6b21a8", "border": "#f0abfc"}
            }
        else:
            node_color = {
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": "#3b82f6", "border": "#60a5fa"}
            }

        nodes_payload.append({
            "id": p["id"],
            "label": p["label"],
            "size": p["size"],
            "color": node_color,
            "confBadge": p["conf_badge"],
            "borderWidth": 4 if p["is_seed"] else 2.5,
            "shape": "dot",
            "font": {
                "color": "#0f172a",
                "size": 14,
                "face": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                "strokeWidth": 4.0,
                "strokeColor": "#ffffff",
                "bold": True,
                "vadjust": 5
            }
        })

        meta_payload.append([
            p["id"],
            {
                "title": p["title"],
                "authors": p["authors"],
                "year": p["year"],
                "citations": p["citations"],
                "venue": p["venue"],
                "conf_badge": p["conf_badge"],
                "abstract": p["abstract"],
                "pdf_url": p["pdf_url"],
                "is_seed": p["is_seed"]
            }
        ])

    edges_payload = []
    edge_set = set()

    for i in range(n):
        peers = np.argsort(stretched_sim[i])[::-1]
        connected = 0
        for j in peers:
            if i == j:
                continue

            sim = float(stretched_sim[i, j])
            pair = tuple(sorted([parsed[i]["id"], parsed[j]["id"]]))
            if pair not in edge_set:
                edge_set.add(pair)

                u_id = parsed[i]["id"]
                v_id = parsed[j]["id"]
                u_is_seed = parsed[i]["is_seed"]
                v_is_seed = parsed[j]["is_seed"]

                # Use explicit is_seed identity instead of fragile index assumptions
                u_cites_v = (u_is_seed and v_id in seed_refs_ids) or (v_id in parsed[i]["refs"])
                v_cites_u = (v_is_seed and u_id in seed_refs_ids) or (u_is_seed and v_id in seed_cites_ids) or (u_id in parsed[j]["refs"])

                spring_len = int(18 + (1.0 - sim) * 62)

                if u_cites_v and not v_cites_u:
                    edges_payload.append({
                        "id": f"{u_id}__{v_id}",
                        "from": u_id,
                        "to": v_id,
                        "type": "cites",
                        "length": spring_len,
                        "width": 2.2,
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
                        "dashes": False,
                        "color": {"color": "rgba(100, 116, 139, 0.75)"}
                    })
                elif v_cites_u and not u_cites_v:
                    edges_payload.append({
                        "id": f"{v_id}__{u_id}",
                        "from": v_id,
                        "to": u_id,
                        "type": "cites",
                        "length": spring_len,
                        "width": 2.2,
                        "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
                        "dashes": False,
                        "color": {"color": "rgba(100, 116, 139, 0.75)"}
                    })
                else:
                    edges_payload.append({
                        "id": f"{u_id}__{v_id}",
                        "from": u_id,
                        "to": v_id,
                        "type": "similarity",
                        "length": spring_len,
                        "width": max(1.2, sim * 3.6),
                        "arrows": {"to": {"enabled": False}},
                        "dashes": [4, 4],
                        "color": {"color": "rgba(203, 213, 225, 0.75)"}
                    })

            connected += 1
            if connected >= 3:
                break

    return nodes_payload, edges_payload, meta_payload, min_y, max_y


# =============================================================================
# HARVESTER (DEDUPLICATED & BROAD HORIZON)
# =============================================================================
def generate_dual_literature_graphs(query: str):
    seed_id = resolve_paper(query)
    if not seed_id:
        print("[!] Could not resolve paper. Please check title or identifier.")
        sys.exit(1)

    print(f"[*] Fetching seed metadata for: {seed_id}")
    seed_fields = (
        "paperId,title,year,citationCount,abstract,authors,openAccessPdf,venue,publicationVenue,"
        "references.paperId,references.title,references.year,references.citationCount,references.authors,references.venue,"
        "citations.paperId,citations.title,citations.year,citations.citationCount,citations.authors,citations.venue"
    )
    seed = client.get(f"/paper/{seed_id}", params={"fields": seed_fields})
    if not seed or "paperId" not in seed:
        print("[!] Seed retrieval failed.")
        sys.exit(1)

    seed_norm_title = normalize_title(seed.get("title", ""))
    seed_refs = seed.get("references") or []
    seed_cites = seed.get("citations") or []
    seed_refs_ids = {r["paperId"] for r in seed_refs if r and r.get("paperId")}
    seed_cites_ids = {c["paperId"] for c in seed_cites if c and c.get("paperId")}

    domain_query = extract_clean_domain_query(seed.get("title", ""))
    print(f"[+] Canonical domain mechanism query: '{domain_query}'")

    # Helper: add to pool with strict title-level deduplication
    def safe_add_paper(pool_dict, title_map, p_obj):
        if not p_obj or not p_obj.get("paperId"):
            return
        pid = p_obj["paperId"]
        if pid == seed_id:
            return
        norm_t = normalize_title(p_obj.get("title", ""))
        if not norm_t or norm_t == seed_norm_title:
            return
        if norm_t in title_map:
            exist_pid = title_map[norm_t]
            if not pool_dict[exist_pid].get("venue") and p_obj.get("venue"):
                pool_dict[exist_pid]["venue"] = p_obj.get("venue")
            if not pool_dict[exist_pid].get("authors") and p_obj.get("authors"):
                pool_dict[exist_pid]["authors"] = p_obj.get("authors")
            return
        title_map[norm_t] = pid
        pool_dict[pid] = p_obj

    # -------------------------------------------------------------------------
    # GRAPH 1: BLEEDING EDGE (CONTEMPORARIES + DIRECT SIBLINGS & PRECURSORS)
    # -------------------------------------------------------------------------
    print("\n[1/2] Harvesting Bleeding-Edge Cluster (Contemporaries & Baselines)...")
    recent_cand_pool = {}
    recent_title_map = {}

    # 1. Forward citations
    for c in seed_cites:
        safe_add_paper(recent_cand_pool, recent_title_map, c)

    # 2. Direct References
    for r in seed_refs:
        safe_add_paper(recent_cand_pool, recent_title_map, r)

    # 3. SPECTER vector recommendations
    rec_res = client.get(f"{REC_URL}/papers/forpaper/{seed_id}?limit=100&fields=paperId,title")
    if rec_res and rec_res.get("recommendedPapers"):
        for rp in rec_res["recommendedPapers"]:
            pid = rp.get("paperId")
            norm_t = normalize_title(rp.get("title", ""))
            if pid and pid != seed_id and norm_t not in recent_title_map:
                recent_cand_pool[pid] = rp
                recent_title_map[norm_t] = pid

    # 4. Same-domain sibling search
    if domain_query:
        d_res = client.get("/paper/search", params={"query": domain_query, "limit": 25, "fields": "paperId,title,year,citationCount,authors,venue"})
        if d_res and d_res.get("data"):
            for p in d_res["data"]:
                safe_add_paper(recent_cand_pool, recent_title_map, p)

    # 5. Author lineage search
    authors = seed.get("authors") or []
    if authors and authors[0].get("authorId") and domain_query:
        first_author_id = authors[0]["authorId"]
        a_res = client.get(f"/author/{first_author_id}/papers", params={"fields": "paperId,title,year,citationCount,venue", "limit": 15})
        if a_res and a_res.get("data"):
            for p in a_res["data"]:
                p_title = p.get("title", "").lower()
                if any(w in p_title for w in domain_query.split()):
                    safe_add_paper(recent_cand_pool, recent_title_map, p)

    # Batch hydrate missing metadata
    ids_to_hydrate = [pid for pid, p in recent_cand_pool.items() if not p.get("abstract") or not p.get("authors")]
    if ids_to_hydrate:
        batch_res = client.post("/paper/batch", json_data={"ids": ids_to_hydrate[:50]}, params={"fields": "paperId,title,year,citationCount,abstract,authors,venue,openAccessPdf"})
        if batch_res and isinstance(batch_res, list):
            for p in batch_res:
                if p and p.get("paperId") in recent_cand_pool:
                    recent_cand_pool[p["paperId"]].update(p)

    recent_papers = [seed] + list(recent_cand_pool.values())[:34]
    r_nodes, r_edges, r_meta, r_min_y, r_max_y = process_subgraph(recent_papers, seed_id, seed_refs_ids, seed_cites_ids)
    print(f"[+] Bleeding Edge graph ready: {len(r_nodes)} unique nodes, {len(r_edges)} edges.")

    # -------------------------------------------------------------------------
    # GRAPH 2: FOUNDATIONAL ROOTS (DIRECT PEDIGREE + DOMAIN ARCHITECTURE ANCESTORS)
    # -------------------------------------------------------------------------
    print("\n[2/2] Harvesting Foundational Roots Graph (Ancestors & Lineage)...")
    foundational_pool = {}
    foundational_title_map = {}

    for r in seed_refs:
        safe_add_paper(foundational_pool, foundational_title_map, r)

    if len(foundational_pool) < 30 and domain_query:
        s_res = client.get("/paper/search", params={
            "query": domain_query,
            "limit": 30,
            "fields": "paperId,title,year,citationCount,authors,venue,abstract"
        })
        if s_res and s_res.get("data"):
            for item in s_res["data"]:
                safe_add_paper(foundational_pool, foundational_title_map, item)

    f_ids_to_hydrate = [pid for pid, p in foundational_pool.items() if not p.get("abstract") or not p.get("authors")]
    if f_ids_to_hydrate:
        batch_res = client.post("/paper/batch", json_data={"ids": f_ids_to_hydrate[:50]}, params={"fields": "paperId,title,year,citationCount,abstract,authors,venue,openAccessPdf"})
        if batch_res and isinstance(batch_res, list):
            for p in batch_res:
                if p and p.get("paperId") in foundational_pool:
                    foundational_pool[p["paperId"]].update(p)

    f_candidates = list(foundational_pool.values())
    f_candidates.sort(key=lambda x: (
        (x.get("paperId") in seed_refs_ids) * 1000 +
        min(int(x.get("citationCount") or 0), 800)
    ), reverse=True)

    foundational_papers = [seed] + f_candidates[:34]
    f_nodes, f_edges, f_meta, f_min_y, f_max_y = process_subgraph(foundational_papers, seed_id, seed_refs_ids, seed_cites_ids)
    print(f"[+] Foundational Roots graph ready: {len(f_nodes)} unique nodes, {len(f_edges)} edges (Years: {f_min_y} - {f_max_y}).")

    return {
        "seed_id": seed_id,
        "recent": {"nodes": r_nodes, "edges": r_edges, "meta": r_meta, "min_y": r_min_y, "max_y": r_max_y},
        "foundational": {"nodes": f_nodes, "edges": f_edges, "meta": f_meta, "min_y": f_min_y, "max_y": f_max_y}
    }


def safe_json_for_script(obj) -> str:
    """Safely escapes json to prevent premature script termination from string sequences like </script>."""
    return json.dumps(obj).replace("</", "<\\/")


def export_dual_html(graph_data, output_path="graph.html"):
    seed_id = graph_data["seed_id"]
    r_data = graph_data["recent"]
    f_data = graph_data["foundational"]

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Connected Literature Graph</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      display: flex;
      width: 100vw;
      height: 100vh;
      background: #ffffff;
      color: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }
    #viewport-container {
      flex: 1;
      height: 100%;
      position: relative;
      background-color: #fcfcfd;
      background-image: radial-gradient(#e2e8f0 1.2px, transparent 1.2px);
      background-size: 24px 24px;
    }
    .network-canvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
    }
    #graph-switcher {
      position: absolute;
      top: 20px;
      left: 24px;
      z-index: 20;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border: 1px solid #cbd5e1;
      border-radius: 9999px;
      padding: 4px;
      display: flex;
      gap: 6px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
    }
    .graph-tab-btn {
      border: none;
      background: transparent;
      padding: 8px 18px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 700;
      color: #64748b;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .graph-tab-btn:hover { color: #0f172a; }
    .graph-tab-btn.active {
      background: #0f172a;
      color: #ffffff;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15);
    }
    .tab-badge {
      padding: 2px 6px;
      border-radius: 9999px;
      background: rgba(100, 116, 139, 0.15);
      font-size: 10px;
    }
    .graph-tab-btn.active .tab-badge {
      background: rgba(255, 255, 255, 0.25);
    }
    #sidebar {
      width: 440px;
      height: 100%;
      background: #ffffff;
      border-left: 1px solid #e2e8f0;
      padding: 28px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: -4px 0 20px rgba(0, 0, 0, 0.03);
      z-index: 30;
    }
    .badge {
      display: inline-block;
      width: fit-content;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      background: #f1f5f9;
      color: #475569;
    }
    .badge-seed { background: #fdf4ff; color: #7e22ce; border: 1px solid #f0abfc; }
    h2 { font-size: 1.18rem; line-height: 1.4; color: #0f172a; font-weight: 700; }
    .authors-container { font-size: 0.85rem; color: #64748b; line-height: 1.5; }
    .authors-toggle {
      display: inline;
      color: #0284c7;
      font-weight: 700;
      font-size: 11px;
      cursor: pointer;
      user-select: none;
    }
    .authors-dropdown {
      margin-top: 6px;
      padding: 8px 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      font-size: 11.5px;
      color: #475569;
      max-height: 110px;
      overflow-y: auto;
      line-height: 1.6;
    }
    .venue-tagline {
      font-size: 0.82rem;
      font-weight: 600;
      color: #475569;
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      margin-top: -6px;
      margin-bottom: 2px;
    }
    .venue-pill {
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      padding: 2px 8px;
      border-radius: 4px;
      color: #334155;
      font-size: 0.78rem;
    }
    .conf-pill {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 700;
      white-space: nowrap;
    }
    .meta-box {
      display: flex;
      gap: 24px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      padding: 12px 16px;
      border-radius: 8px;
    }
    .meta-val { font-size: 1.15rem; font-weight: 700; color: #0f172a; display: block; }
    .meta-lbl { color: #64748b; font-size: 0.72rem; text-transform: uppercase; font-weight: 600; }
    #legend {
      position: absolute;
      bottom: 24px;
      left: 24px;
      z-index: 20;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      padding: 16px 20px;
      border-radius: 10px;
      border: 1px solid #e2e8f0;
      box-shadow: 0 4px 12px rgba(0,0,0,0.05);
      font-size: 11.5px;
      pointer-events: none;
      display: flex;
      flex-direction: column;
      gap: 7px;
    }
    .gradient-bar {
      width: 180px;
      height: 10px;
      border-radius: 5px;
      background: linear-gradient(to right, rgb(212,224,224), rgb(107,153,158), rgb(38,75,82));
      border: 1px solid #cbd5e1;
      margin: 4px 0 2px 0;
    }
    .gradient-labels { display: flex; justify-content: space-between; font-size: 10.5px; color: #64748b; font-weight: 700; }
    .edge-legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: #475569;
    }
    .legend-arrow {
      width: 24px;
      height: 2px;
      background: #64748b;
      position: relative;
    }
    .legend-arrow::after {
      content: '';
      position: absolute;
      right: 0;
      top: -3px;
      width: 0;
      height: 0;
      border-top: 4px solid transparent;
      border-bottom: 4px solid transparent;
      border-left: 6px solid #64748b;
    }
    .legend-dash {
      width: 24px;
      height: 0;
      border-top: 2px dashed #cbd5e1;
    }
    .btn {
      padding: 10px 16px;
      border-radius: 6px;
      text-decoration: none;
      font-size: 0.85rem;
      font-weight: 600;
      text-align: center;
      display: inline-block;
    }
    .btn-pdf { background: #0f172a; color: #fff; }
    .btn-s2 { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
  </style>
</head>
<body>
  <div id="viewport-container">
    <div id="graph-switcher">
      <button id="tab-recent" class="graph-tab-btn active" onclick="activateGraph('recent')">
        Bleeding Edge (Contemporaries) <span class="tab-badge">__COUNT_RECENT__</span>
      </button>
      <button id="tab-foundational" class="graph-tab-btn" onclick="activateGraph('foundational')">
        Foundational Roots (Ancestors) <span class="tab-badge">__COUNT_FOUNDATIONAL__</span>
      </button>
    </div>

    <div id="legend">
      <div>
        <div style="font-weight: 700; color: #0f172a;">Publication Year</div>
        <div class="gradient-bar"></div>
        <div class="gradient-labels"><span id="lbl-min-y">-</span><span id="lbl-max-y">-</span></div>
      </div>
      <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #7e22ce;">
        <span style="width: 12px; height: 12px; border-radius: 50%; background: #7e22ce; border: 2px solid #e879f9; display: inline-block;"></span>
        <span>Origin Seed Paper</span>
      </div>
      <div class="edge-legend-row">
        <div class="legend-arrow"></div>
        <span><b>Direct Citation</b> (A cites B)</span>
      </div>
      <div class="edge-legend-row">
        <div class="legend-dash"></div>
        <span><b>Similarity Link</b> (Co-citations / Text)</span>
      </div>
      <div style="margin-top: 5px; padding-top: 6px; border-top: 1px solid #e2e8f0; display: flex; flex-direction: column; gap: 4px;">
        <div style="font-weight: 700; color: #0f172a; font-size: 11px;">Conference Tier Badge</div>
        <div class="edge-legend-row">
          <span style="width: 11px; height: 11px; border-radius: 50%; background: #ef4444; border: 2px solid #ffffff; box-shadow: 0 0 0 1px #cbd5e1; display: inline-block;"></span>
          <span><b>Top Arch:</b> ISCA, MICRO, ASPLOS, HPCA</span>
        </div>
        <div class="edge-legend-row">
          <span style="width: 11px; height: 11px; border-radius: 50%; background: #a855f7; border: 2px solid #ffffff; box-shadow: 0 0 0 1px #cbd5e1; display: inline-block;"></span>
          <span><b>Top Systems/CS:</b> SOSP, OSDI, NSDI, SIGCOMM...</span>
        </div>
        <div class="edge-legend-row">
          <span style="width: 11px; height: 11px; border-radius: 50%; background: #f97316; border: 2px solid #ffffff; box-shadow: 0 0 0 1px #cbd5e1; display: inline-block;"></span>
          <span><b>Premier Arch/Systems:</b> DAC, EuroSys, ATC, FAST...</span>
        </div>
        <div class="edge-legend-row">
          <span style="width: 11px; height: 11px; border-radius: 50%; background: #94a3b8; border: 2px solid #ffffff; box-shadow: 0 0 0 1px #cbd5e1; display: inline-block;"></span>
          <span><b>Preprint / In-Flight:</b> arXiv, bioRxiv (Not yet evaluated)</span>
        </div>
        <div class="edge-legend-row" style="opacity: 0.85;">
          <span style="width: 11px; height: 11px; border-radius: 50%; background: transparent; border: 1.5px dashed #94a3b8; display: inline-block;"></span>
          <span><b>No Badge:</b> Peer-reviewed in other/lower-tier venues</span>
        </div>
      </div>
    </div>

    <div id="canvas-recent" class="network-canvas"></div>
    <div id="canvas-foundational" class="network-canvas" style="display: none;"></div>
  </div>

  <div id="sidebar">
    <div id="placeholder" style="margin: auto; text-align: center; color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">
      Click any node to pin it, or hover to preview.
    </div>
    <div id="details" style="display: none; flex-direction: column; gap: 16px;">
      <span id="p-badge" class="badge">Paper</span>
      <h2 id="p-title"></h2>
      <div id="p-authors" class="authors-container"></div>
      <div id="p-venue-line" class="venue-tagline"></div>
      <div class="meta-box">
        <div><span id="p-year" class="meta-val">-</span><span class="meta-lbl">Year</span></div>
        <div><span id="p-citations" class="meta-val">-</span><span class="meta-lbl">Citations</span></div>
      </div>
      <div style="display: flex; gap: 8px;">
        <a id="p-pdf" href="#" target="_blank" class="btn btn-pdf" style="flex:1;">View PDF ↗</a>
        <a id="p-s2" href="#" target="_blank" class="btn btn-s2" style="flex:1;">Semantic Scholar ↗</a>
      </div>
      <p id="p-desc" style="font-size: 0.88rem; line-height: 1.65; color: #334155; max-height: 380px; overflow-y: auto;"></p>
    </div>
  </div>

  <script>
    const SEED_ID = "__SEED_ID__";

    const graphs = {
      recent: {
        nodes: __R_NODES__,
        edges: __R_EDGES__,
        meta: new Map(__R_META__),
        min_y: __R_MIN_Y__,
        max_y: __R_MAX_Y__,
        domId: 'canvas-recent',
        pinned: SEED_ID,
        network: null,
        nodesDs: null,
        edgesDs: null,
        adjList: {}
      },
      foundational: {
        nodes: __F_NODES__,
        edges: __F_EDGES__,
        meta: new Map(__F_META__),
        min_y: __F_MIN_Y__,
        max_y: __F_MAX_Y__,
        domId: 'canvas-foundational',
        pinned: SEED_ID,
        network: null,
        nodesDs: null,
        edgesDs: null,
        adjList: {}
      }
    };

    let activeGraphKey = 'recent';

    function initNetworkInstance(gKey) {
      const g = graphs[gKey];
      g.nodesDs = new vis.DataSet(g.nodes);
      g.edgesDs = new vis.DataSet(g.edges);

      g.nodes.forEach(n => { g.adjList[n.id] = []; });
      g.edges.forEach(e => {
        g.adjList[e.from].push({ neighbor: e.to, edgeId: e.id, isOutgoingCite: (e.type === 'cites') });
        g.adjList[e.to].push({ neighbor: e.from, edgeId: e.id, isIncomingCite: (e.type === 'cites') });
      });

      const container = document.getElementById(g.domId);
      g.network = new vis.Network(container, { nodes: g.nodesDs, edges: g.edgesDs }, {
        physics: {
          solver: 'forceAtlas2Based',
          forceAtlas2Based: {
            gravitationalConstant: -110,
            centralGravity: 0.007,
            springConstant: 0.16,
            damping: 0.65,
            avoidOverlap: 0.25
          },
          stabilization: { enabled: true, iterations: 260 }
        },
        interaction: { hover: true, hoverConnectedEdges: false, selectConnectedEdges: false }
      });

      g.network.once('stabilizationIterationsDone', function () {
        g.network.setOptions({ physics: false });
        g.network.fit();
        if (activeGraphKey === gKey) {
          updateLegend(gKey);
          updateDetails(g.pinned, gKey);
          applyHighlight(g.pinned, gKey);
        }
      });

      // OVERLAY CONFERENCE BADGE
      g.network.on("afterDrawing", function (ctx) {
        const positions = g.network.getPositions();
        const nodesList = g.nodes;

        for (let i = 0; i < nodesList.length; i++) {
          const n = nodesList[i];
          if (n.confBadge && positions[n.id]) {
            const pos = positions[n.id];
            const r = n.size;

            const badgeX = pos.x + r * 0.7071;
            const badgeY = pos.y + r * 0.7071;
            const badgeR = Math.max(5.5, Math.min(12.5, r * 0.35));

            ctx.beginPath();
            ctx.arc(badgeX, badgeY, badgeR + 2.5, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffffff";
            ctx.fill();

            ctx.beginPath();
            ctx.arc(badgeX, badgeY, badgeR, 0, 2 * Math.PI);
            ctx.fillStyle = n.confBadge.color;
            ctx.fill();
          }
        }
      });

      g.network.on("hoverNode", function (params) {
        updateDetails(params.node, gKey);
        applyHighlight(params.node, gKey);
      });

      g.network.on("blurNode", function () {
        if (g.pinned) {
          updateDetails(g.pinned, gKey);
          applyHighlight(g.pinned, gKey);
        } else {
          clearHighlight(gKey);
          document.getElementById("placeholder").style.display = "block";
          document.getElementById("details").style.display = "none";
        }
      });

      g.network.on("click", function (params) {
        if (params.nodes.length > 0) {
          g.pinned = params.nodes[0];
          updateDetails(g.pinned, gKey);
          applyHighlight(g.pinned, gKey);
        } else {
          g.pinned = null;
          clearHighlight(gKey);
          document.getElementById("placeholder").style.display = "block";
          document.getElementById("details").style.display = "none";
        }
      });

      let zoomRaf = null;
      g.network.on("zoom", function () {
        if (zoomRaf) cancelAnimationFrame(zoomRaf);
        zoomRaf = requestAnimationFrame(function () {
          if (g.pinned) applyHighlight(g.pinned, gKey);
          else clearHighlight(gKey);
        });
      });
    }

    function getPathToSeed(startNodeId, gKey) {
      if (startNodeId === SEED_ID) return { nodes: [SEED_ID], edges: [] };
      const g = graphs[gKey];
      const queue = [[startNodeId]];
      const visited = new Set([startNodeId]);
      const edgeMap = {};

      while (queue.length > 0) {
        const path = queue.shift();
        const curr = path[path.length - 1];
        if (curr === SEED_ID) {
          const pathEdges = [];
          for (let i = 0; i < path.length - 1; i++) {
            pathEdges.push(edgeMap[`${path[i]}->${path[i+1]}`]);
          }
          return { nodes: path, edges: pathEdges };
        }
        for (const conn of g.adjList[curr] || []) {
          if (!visited.has(conn.neighbor)) {
            visited.add(conn.neighbor);
            edgeMap[`${curr}->${conn.neighbor}`] = conn.edgeId;
            edgeMap[`${conn.neighbor}->${curr}`] = conn.edgeId;
            queue.push([...path, conn.neighbor]);
          }
        }
      }
      return { nodes: [startNodeId], edges: [] };
    }

    function getZoomScaledFont(scale, active = false) {
      const screenPx = active ? 15.0 : 13.0;
      const canvasPx = Math.max(6, Math.round(screenPx / (scale || 1.0)));
      const strokeW = Math.max(1.5, Math.round(4.0 / (scale || 1.0)));
      const vadjust = Math.max(2, Math.round(5.0 / (scale || 1.0)));
      return { size: canvasPx, strokeWidth: strokeW, vadjust: vadjust };
    }

    function applyHighlight(targetId, gKey) {
      const g = graphs[gKey];
      const directConns = g.adjList[targetId] || [];
      const directNeighbors = new Set(directConns.map(c => c.neighbor));

      const outgoingEdges = new Set();
      const incomingEdges = new Set();
      const similarityEdges = new Set();

      directConns.forEach(c => {
        if (c.isOutgoingCite) outgoingEdges.add(c.edgeId);
        else if (c.isIncomingCite) incomingEdges.add(c.edgeId);
        else similarityEdges.add(c.edgeId);
      });

      const pathToSeed = getPathToSeed(targetId, gKey);
      const seedPathNodes = new Set(pathToSeed.nodes);
      const seedPathEdges = new Set(pathToSeed.edges);

      const scale = g.network.getScale() || 1.0;
      const fontActive = getZoomScaledFont(scale, true);
      const fontNormal = getZoomScaledFont(scale, false);

      g.nodesDs.update(g.nodes.map(n => {
        const isTarget = n.id === targetId;
        const isSeed = n.id === SEED_ID;
        const isPath = seedPathNodes.has(n.id);
        const isNeighbor = directNeighbors.has(n.id);
        const active = isTarget || isSeed || isPath || isNeighbor;
        const fontSpec = active ? fontActive : fontNormal;

        return {
          id: n.id,
          color: {
            background: n.color.background,
            border: isTarget ? "#0284c7" : (isSeed ? "#e879f9" : (isPath ? "#c026d3" : (isNeighbor ? "#38bdf8" : "#ffffff")))
          },
          borderWidth: isTarget ? 5.0 : (isSeed ? 4.5 : (active ? 3.5 : 2.5)),
          font: {
            size: fontSpec.size,
            face: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
            strokeWidth: fontSpec.strokeWidth,
            strokeColor: "#ffffff",
            color: active ? "#0f172a" : "#475569",
            bold: true,
            vadjust: fontSpec.vadjust
          }
        };
      }));

      g.edgesDs.update(g.edges.map(e => {
        if (seedPathEdges.has(e.id)) {
          return { id: e.id, width: 4.0, color: { color: "#c026d3", opacity: 1.0 } };
        } else if (outgoingEdges.has(e.id)) {
          return { id: e.id, width: 3.2, color: { color: "#f59e0b", opacity: 1.0 } };
        } else if (incomingEdges.has(e.id)) {
          return { id: e.id, width: 3.2, color: { color: "#06b6d4", opacity: 1.0 } };
        } else if (similarityEdges.has(e.id)) {
          return { id: e.id, width: 2.0, color: { color: "#64748b", opacity: 0.95 } };
        } else {
          return { id: e.id, width: 1.1, color: { color: "rgba(203, 213, 225, 0.45)" } };
        }
      }));
    }

    function clearHighlight(gKey) {
      const g = graphs[gKey];
      const scale = g.network.getScale() || 1.0;
      const fontNormal = getZoomScaledFont(scale, false);

      g.nodesDs.update(g.nodes.map(n => ({
        id: n.id,
        color: n.color,
        borderWidth: n.borderWidth,
        font: {
          size: fontNormal.size,
          face: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
          strokeWidth: fontNormal.strokeWidth,
          strokeColor: "#ffffff",
          color: "#0f172a",
          bold: true,
          vadjust: fontNormal.vadjust
        }
      })));
      g.edgesDs.update(g.edges.map(e => ({
        id: e.id,
        width: e.width,
        color: e.color
      })));
    }

    function renderAuthors(authorsList) {
      if (!authorsList || authorsList.length === 0) return "Unknown Authors";
      if (authorsList.length <= 2) return authorsList.join(", ");

      const visible = authorsList.slice(0, 2).join(", ");
      const extraCount = authorsList.length - 2;
      const hiddenAuthors = authorsList.slice(2).join(", ");

      return visible + `
        <details style="display:inline; margin-left: 4px;">
          <summary class="authors-toggle">+ ${extraCount} more</summary>
          <div class="authors-dropdown">${hiddenAuthors}</div>
        </details>
      `;
    }

    function updateDetails(paperId, gKey) {
      const g = graphs[gKey];
      const paper = g.meta.get(paperId);
      if (!paper) return;

      document.getElementById("placeholder").style.display = "none";
      const details = document.getElementById("details");
      details.style.display = "flex";

      document.getElementById("p-title").innerText = paper.title;
      document.getElementById("p-authors").innerHTML = renderAuthors(paper.authors);

      const venueEl = document.getElementById("p-venue-line");
      const venueParts = [];
      if (paper.year) venueParts.push(paper.year);
      if (paper.venue && paper.venue.trim().length > 0) {
        venueParts.push(`<span class="venue-pill">${paper.venue}</span>`);
      }
      if (paper.conf_badge) {
        const badgeHtml = `
          <span class="conf-pill" style="background: ${paper.conf_badge.bg}; color: ${paper.conf_badge.color}; border: 1px solid ${paper.conf_badge.border};">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background: ${paper.conf_badge.color}; margin-right:4px;"></span>
            ${paper.conf_badge.abbr} • ${paper.conf_badge.tier_name}
          </span>
        `;
        venueParts.push(badgeHtml);
      }
      venueEl.innerHTML = venueParts.join('<span style="color:#94a3b8;">•</span>');

      document.getElementById("p-year").innerText = paper.year;
      document.getElementById("p-citations").innerText = Number(paper.citations).toLocaleString();
      document.getElementById("p-desc").innerText = paper.abstract;
      document.getElementById("p-s2").href = "https://www.semanticscholar.org/paper/" + paperId;

      const pdfBtn = document.getElementById("p-pdf");
      if (paper.pdf_url) {
        pdfBtn.style.display = "inline-block";
        pdfBtn.href = paper.pdf_url;
      } else {
        pdfBtn.style.display = "none";
      }

      const badge = document.getElementById("p-badge");
      if (paper.is_seed) {
        badge.className = "badge badge-seed";
        badge.innerText = "Origin Seed Paper";
      } else if (gKey === 'recent') {
        badge.className = "badge";
        badge.innerText = "Contemporary Literature";
      } else {
        badge.className = "badge";
        badge.innerText = "Foundational Literature";
      }
    }

    function updateLegend(gKey) {
      document.getElementById('lbl-min-y').innerText = graphs[gKey].min_y;
      document.getElementById('lbl-max-y').innerText = graphs[gKey].max_y;
    }

    window.activateGraph = function(gKey) {
      if (activeGraphKey === gKey) return;
      activeGraphKey = gKey;

      document.querySelectorAll('.graph-tab-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('tab-' + gKey).classList.add('active');

      document.getElementById('canvas-recent').style.display = (gKey === 'recent') ? 'block' : 'none';
      document.getElementById('canvas-foundational').style.display = (gKey === 'foundational') ? 'block' : 'none';

      updateLegend(gKey);
      const g = graphs[gKey];
      if (g.network) {
        g.network.redraw();
        updateDetails(g.pinned, gKey);
        applyHighlight(g.pinned, gKey);
      }
    };

    initNetworkInstance('recent');
    initNetworkInstance('foundational');
  </script>
</body>
</html>"""

    html = (
        template
        .replace("__SEED_ID__", str(seed_id))
        .replace("__R_NODES__", safe_json_for_script(r_data["nodes"]))
        .replace("__R_EDGES__", safe_json_for_script(r_data["edges"]))
        .replace("__R_META__", safe_json_for_script(r_data["meta"]))
        .replace("__R_MIN_Y__", str(r_data["min_y"]))
        .replace("__R_MAX_Y__", str(r_data["max_y"]))
        .replace("__COUNT_RECENT__", str(len(r_data["nodes"])))
        .replace("__F_NODES__", safe_json_for_script(f_data["nodes"]))
        .replace("__F_EDGES__", safe_json_for_script(f_data["edges"]))
        .replace("__F_META__", safe_json_for_script(f_data["meta"]))
        .replace("__F_MIN_Y__", str(f_data["min_y"]))
        .replace("__F_MAX_Y__", str(f_data["max_y"]))
        .replace("__COUNT_FOUNDATIONAL__", str(len(f_data["nodes"])))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[+] Dual-graph literature interface saved: {output_path}")

    # Cross-platform browser opening
    try:
        resolved_uri = Path(output_path).resolve().as_uri()
        webbrowser.open(resolved_uri)
    except Exception:
        print(f"Open manually in your web browser: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        print("\n=======================================================")
        print(" Connected Papers CLI - Conference Tier Engine")
        print("=======================================================")
        if not API_KEY:
            print("[!] Note: S2_API_KEY environment variable not found.")
            print("    Running in public unauthenticated mode (rate limits apply).")
            print("    To add a key, add S2_API_KEY=your_key in .env\n")
        try:
            query = input("[?] Paste paper title, DOI, or arXiv URL: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    while not query:
        try:
            query = input("[?] Please paste a valid paper title or identifier: ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    graph_bundle = generate_dual_literature_graphs(query)
    export_dual_html(graph_bundle)
