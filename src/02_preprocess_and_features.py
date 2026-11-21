#!/usr/bin/env python3
"""Clean presidential speeches, balance sample, and extract leakage-free stylometric features."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

SEED = 42
MIN_WORDS = 120
N_PER_PRESIDENT = 300

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BASE = PROJECT_ROOT / "data" / "raw" / "base e códigos"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

PRESIDENT_MAP = {
    "00 Sarney": "Sarney",
    "01 Collor": "Collor",
    "02 Itamar": "Itamar",
    "03 FHC": "FHC",
    "04 Lula": "Lula",
    "05 Dilma": "Dilma",
    "06 Temer": "Temer",
    "07 Bolsonaro": "Bolsonaro",
}

PRESIDENT_ORDER = ["Sarney", "FHC", "Lula", "Dilma", "Temer", "Bolsonaro"]

MONTHS_RE = (
    r"(?:janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|"
    r"setembro|outubro|novembro|dezembro)"
)

PT_STOPWORDS = {
    "de", "que", "o", "a", "e", "do", "da", "em", "para", "com", "não", "nao",
    "é", "os", "as", "um", "uma", "no", "na", "por", "se", "mais", "como", "dos", "das",
}
EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "for", "that", "with", "this", "are", "be", "we", "our",
}

FIRST_SG = {"eu", "me", "mim", "comigo", "meu", "minha", "meus", "minhas"}
FIRST_PL = {"nós", "nos", "conosco", "nosso", "nossa", "nossos", "nossas"}
VOCATIVES = {
    "senhor", "senhora", "senhores", "senhoras", "vossa", "excelência",
    "companheiro", "companheiros", "companheira", "companheiras",
    "amigos", "amigas", "amigo", "amiga", "brasileiros", "brasileiras",
    "cidadãos", "cidadãs", "povo", "gente", "pessoal",
}
ORALITY_WORDS = {
    "olha", "olhe", "então", "aí", "né", "tá", "pra", "pro", "gente", "pessoal",
    "porque", "coisa", "coisas", "tudo", "aqui", "lá", "bom", "veja", "enfim", "aliás",
}
FORMAL_CONNECTORS = {
    "todavia", "contudo", "entretanto", "portanto", "ademais", "outrossim",
    "destarte", "mediante", "acerca", "conquanto", "porquanto", "mormente",
    "igualmente", "sobretudo", "ora", "outrora", "deveras", "não obstante",
}

# Explicit function-word vocabulary (~200 closed-class words, zero topical terms)
FUNCTION_WORDS = sorted(
    {
        "a", "à", "ao", "aos", "as", "às", "ante", "após", "até", "com", "contra",
        "da", "das", "de", "dela", "delas", "dele", "deles", "desde", "do", "dos",
        "e", "ela", "elas", "ele", "eles", "em", "entre", "essa", "essas", "esse",
        "esses", "esta", "estas", "este", "estes", "eu", "isso", "isto", "aquela",
        "aquelas", "aquele", "aqueles", "aquilo", "lhe", "lhes", "me", "meu", "meus",
        "mim", "minha", "minhas", "na", "nas", "nem", "no", "nos", "nós", "nossa",
        "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela",
        "pelas", "pelo", "pelos", "perante", "por", "qual", "quais", "quando",
        "quanto", "quanta", "quantos", "quantas", "que", "quem", "se", "sem", "seu",
        "seus", "sob", "sobre", "sua", "suas", "te", "teu", "teus", "tu", "tua",
        "tuas", "um", "uma", "umas", "uns", "você", "vocês", "vos", "vós",
        "comigo", "conosco", "consigo",
        # Auxiliary / copula / light structural verbs
        "é", "era", "eram", "éramos", "foi", "fomos", "for", "fora", "foram",
        "fosse", "fossem", "fui", "seja", "sejam", "ser", "será", "serão", "serei",
        "seremos", "seria", "seriam", "somos", "sou", "são",
        "está", "estão", "estava", "estavam", "estávamos", "esteja", "estejam",
        "esteve", "estive", "estivemos", "estiver", "estiveram", "estivesse",
        "estou", "estamos", "estar", "estará", "estariam",
        "há", "haja", "havemos", "havia", "haviam", "hão", "hei", "houve", "houver",
        "tem", "têm", "temos", "tenha", "tenham", "tenho", "ter", "terá", "terão",
        "terei", "teremos", "teria", "teriam", "teve", "tinha", "tinham", "tínhamos",
        "tive", "tivemos", "tiver", "tiveram", "tivesse", "tivessem",
        # Discourse connectors & closed-class adverbs/quantifiers
        "agora", "ainda", "aí", "ali", "aliás", "antes", "apenas", "aqui", "assim",
        "bem", "cada", "como", "conforme", "contudo", "depois", "embora", "entanto",
        "então", "entretanto", "hoje", "igualmente", "já", "lá", "logo", "mais",
        "mal", "mas", "menos", "mesmo", "mesma", "mesmos", "mesmas", "muita",
        "muitas", "muito", "muitos", "nada", "não", "nenhum", "nenhuma", "nunca",
        "onde", "ora", "outra", "outras", "outro", "outros", "pois", "porém",
        "porque", "portanto", "pouca", "poucas", "pouco", "poucos", "quase",
        "sempre", "sim", "só", "sobretudo", "talvez", "também", "tanta", "tantas",
        "tanto", "tantos", "tão", "toda", "todas", "todo", "todos", "todavia",
        "tudo", "enfim", "ademais", "mediante", "acerca", "enquanto",
    }
)

# Names / temporal identifiers stripped prior to n-gram extraction to prevent leakage
LEAKAGE_PATTERN = re.compile(
    r"\b(?:"
    r"sarney|josé\s+sarney|collor|fernando\s+collor|itamar|itamar\s+franco|"
    r"fhc|fernando\s+henrique|cardoso|lula|luiz\s+inácio|inácio\s+da\s+silva|"
    r"dilma|rousseff|roussef|temer|michel\s+temer|bolsonaro|jair\s+messias|jair|"
    r"presidenta|(?:19|20)\d{2}"
    r")\b",
    flags=re.IGNORECASE,
)

WORD_RE = re.compile(r"[a-zA-ZáàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ]+(?:-[a-zA-Záàâãéêíóôõúüç]+)*")
SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")
VOWEL_GROUP_RE = re.compile(r"[aeiouáàâãéêíóôõúü]+", flags=re.IGNORECASE)


def decode_bytes(raw_bytes: bytes) -> str:
    """Decode text handling both UTF-8 (Lula-Bolsonaro) and CP1252 (Sarney-FHC)."""
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252", errors="replace")


def clean_speech_text(raw_text: str) -> str:
    """Remove editorial headers, chronology blurbs, transcription tags, and normalize dashes."""
    t = raw_text.replace("\xa0", " ").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    # Remove transcription markers like [aplausos], (aplausos), (risos), (falha na gravação), (falha no áudio)
    t = re.sub(
        r"[\[(]\s*(?:aplausos|risos|palmas|incompreensível|falha\s+na\s+gravação|"
        r"falha\s+no\s+áudio|falha\s+técnica|áudio\s+interrompido|cortado|"
        r"inaudível|interrupção|música)[^\]\)]*[\])]",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    # Collapse whitespace before header stripping
    t = re.sub(r"\s+", " ", t).strip()

    # 1) Sarney editorial chronology blocks: "<day/roman/?> de <month> — ... ." possibly followed by ". — ..."
    chrono_matches = list(
        re.finditer(rf"(?:\d+|\?|I{{1,3}})\s+de\s+{MONTHS_RE}\s+[—–]", t[:3200], flags=re.IGNORECASE)
    )
    if chrono_matches:
        last_pos = chrono_matches[-1].end()
        # Check for additional '. — ' entries right after in the chronology block
        extra = list(re.finditer(r"\.\s+[—–]\s+", t[last_pos : last_pos + 1400]))
        if extra:
            last_pos = last_pos + extra[-1].end()
        after = t[last_pos:]
        m_end = re.search(r"\.\s+", after)
        if m_end and m_end.end() < 900:
            t = after[m_end.end() :].strip()

    # 2) Standard date/location/title header at start of speech
    m_date = re.search(
        rf"\b\d{{1,2}}\s+de\s+{MONTHS_RE}(?:\s+de\s+(?:19|20)\d{{2}})?\.?\s*",
        t[:900],
        flags=re.IGNORECASE,
    )
    if m_date and m_date.start() < 750:
        prefix = t[: m_date.start()]
        has_full_year = re.search(r"\b(?:19|20)\d{2}\b", m_date.group(0)) is not None
        geo_suffix = (
            re.search(
                r"(?:DF|SP|RJ|MG|RS|BA|PE|SE|PR|SC|CE|GO|AM|PA|MA|PB|RN|ES|MT|MS|"
                r"Índia|EUA|ONU|Planalto|Brasília)\s*,?\s*$",
                prefix,
                flags=re.IGNORECASE,
            )
            is not None
        )
        if (
            len(prefix) < 140
            or re.search(
                r"(?:Discurso|Pronunciamento|Palavras|Declaração|Palácio|Brasília|BRASÍLIA|"
                r"INAUGURAÇÃO|ABERTURA|SAUDAÇÃO|POSSE|SOLENIDADE|ENTREVISTA|\d{2}/\d{2}/\d{4})",
                prefix,
                flags=re.IGNORECASE,
            )
            or (has_full_year and geo_suffix)
        ):
            t = t[m_date.end() :].strip()

    # 3) Residual leading ALL-CAPS title/city block (common in Sarney / FHC)
    t = t.replace("BRASlLIA", "BRASÍLIA")
    t = re.sub(
        r"^(?:\d{2}/\d{2}/\d{4}\s+)?(?:[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ0-9\-/,:;()\.–]{1,}\s+){3,}(?=(?:[A-ZÁÀÂÃÉÊÍÓÔÕÚÜÇ][a-záàâãéêíóôõúüç]+|[ÉAOE]\s+[a-záàâãéêíóôõúüç]{2,}))",
        "",
        t,
    )

    # Unify typographic dashes so encoding era (em-dash vs en-dash) cannot leak president
    t = t.replace("—", " – ").replace("–", " – ")
    # Clean residual leading ellipsis/punctuation after stripping transcription tags
    t = re.sub(r"^[\.\s…]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_interview(text: str) -> bool:
    """Detect Q&A interview transcripts rather than monologic speeches."""
    hits = len(
        re.findall(
            r"\b(?:Jornalista|Repórter|Entrevistador|Pergunta)\s*:",
            text,
            flags=re.IGNORECASE,
        )
    )
    return hits >= 2


def is_portuguese(words_lower: list[str]) -> bool:
    """Verify text is predominantly Brazilian Portuguese."""
    if not words_lower:
        return False
    pt_frac = sum(1 for w in words_lower if w in PT_STOPWORDS) / len(words_lower)
    en_frac = sum(1 for w in words_lower if w in EN_STOPWORDS) / len(words_lower)
    return pt_frac >= 0.14 and en_frac < 0.08


def count_syllables_pt(word: str) -> int:
    """Approximate Portuguese syllable count via vowel clusters."""
    groups = VOWEL_GROUP_RE.findall(word)
    return max(1, len(groups))


def compute_mtld_one_way(tokens: list[str], threshold: float = 0.72) -> float:
    """Compute one-directional Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010)."""
    if len(tokens) < 10:
        return 0.0
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    for tok in tokens:
        types.add(tok)
        token_count += 1
        ttr = len(types) / token_count
        if ttr <= threshold:
            factors += 1.0
            types.clear()
            token_count = 0
    if token_count > 0:
        ttr = len(types) / token_count
        if ttr < 1.0:
            factors += (1.0 - ttr) / (1.0 - threshold)
    return len(tokens) / factors if factors > 0 else float(len(tokens))


def compute_mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Bidirectional MTLD robust to speech length."""
    fwd = compute_mtld_one_way(tokens, threshold)
    bwd = compute_mtld_one_way(tokens[::-1], threshold)
    return 0.5 * (fwd + bwd)


def compute_mattr(tokens: list[str], window: int = 100) -> float:
    """Moving-Average Type-Token Ratio over fixed sliding window."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    total_ttr = len(counts) / window
    steps = n - window
    for i in range(steps):
        out_tok = tokens[i]
        in_tok = tokens[i + window]
        counts[out_tok] -= 1
        if counts[out_tok] == 0:
            del counts[out_tok]
        counts[in_tok] += 1
        total_ttr += len(counts) / window
    return total_ttr / (steps + 1)


def extract_stylo23(text: str) -> dict[str, float]:
    """Compute 23 artifact-free interpretable stylometric metrics on cleaned speech text."""
    # Sentences
    raw_sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) > 0]
    sent_word_counts = [len(WORD_RE.findall(s)) for s in raw_sents]
    sent_word_counts = [c for c in sent_word_counts if c > 0]
    if not sent_word_counts:
        sent_word_counts = [1]

    words_orig = WORD_RE.findall(text)
    words_lower = [w.lower() for w in words_orig]
    n_words = max(1, len(words_lower))
    n_sents = max(1, len(sent_word_counts))
    scale_1k = 1000.0 / n_words

    avg_sent_len = float(np.mean(sent_word_counts))
    std_sent_len = float(np.std(sent_word_counts))
    short_sent_ratio = float(np.mean([c <= 8 for c in sent_word_counts]))
    long_sent_ratio = float(np.mean([c >= 35 for c in sent_word_counts]))

    word_lens = [len(w) for w in words_lower]
    avg_word_len = float(np.mean(word_lens))
    long_word_ratio = float(np.mean([l >= 7 for l in word_lens]))

    mtld = compute_mtld(words_lower)
    mattr_100 = compute_mattr(words_lower, window=100)

    total_syllables = sum(count_syllables_pt(w) for w in words_lower)
    asw = total_syllables / n_words
    # Flesch reading ease adapted to Brazilian Portuguese (Martins et al., 1996)
    flesch_pt = 248.835 - 1.015 * avg_sent_len - 84.6 * asw

    cnt = Counter(words_lower)
    sg_cnt = sum(cnt[w] for w in FIRST_SG)
    pl_cnt = sum(cnt[w] for w in FIRST_PL)
    first_person_sg_rate = sg_cnt * scale_1k
    first_person_pl_rate = pl_cnt * scale_1k
    first_person_sg_share = sg_cnt / (sg_cnt + pl_cnt + 1e-6)

    vocative_rate = sum(cnt[w] for w in VOCATIVES) * scale_1k
    orality_rate = sum(cnt[w] for w in ORALITY_WORDS) * scale_1k

    # Formal connectors (including multi-word 'não obstante')
    formal_cnt = sum(cnt[w] for w in FORMAL_CONNECTORS if " " not in w)
    formal_cnt += len(re.findall(r"\bnão obstante\b", text.lower()))
    formal_connector_rate = formal_cnt * scale_1k

    exclamation_rate = text.count("!") * scale_1k
    question_rate = text.count("?") * scale_1k
    semicolon_rate = text.count(";") * scale_1k
    colon_rate = text.count(":") * scale_1k
    comma_per_sent = text.count(",") / n_sents

    gerund_rate = sum(1 for w in words_lower if len(w) >= 5 and w.endswith("ndo")) * scale_1k
    adverb_mente_rate = sum(1 for w in words_lower if len(w) >= 7 and w.endswith("mente")) * scale_1k
    nominalization_rate = (
        sum(
            1
            for w in words_lower
            if len(w) >= 6
            and w.endswith(("ção", "ções", "mento", "mentos", "dade", "dades", "ismo", "ismos"))
        )
        * scale_1k
    )

    return {
        "avg_sent_len": avg_sent_len,
        "std_sent_len": std_sent_len,
        "short_sent_ratio": short_sent_ratio,
        "long_sent_ratio": long_sent_ratio,
        "avg_word_len": avg_word_len,
        "long_word_ratio": long_word_ratio,
        "mtld": mtld,
        "mattr_100": mattr_100,
        "flesch_pt": flesch_pt,
        "first_person_sg_rate": first_person_sg_rate,
        "first_person_pl_rate": first_person_pl_rate,
        "first_person_sg_share": first_person_sg_share,
        "vocative_rate": vocative_rate,
        "exclamation_rate": exclamation_rate,
        "question_rate": question_rate,
        "semicolon_rate": semicolon_rate,
        "colon_rate": colon_rate,
        "comma_per_sent": comma_per_sent,
        "orality_rate": orality_rate,
        "formal_connector_rate": formal_connector_rate,
        "gerund_rate": gerund_rate,
        "adverb_mente_rate": adverb_mente_rate,
        "nominalization_rate": nominalization_rate,
    }


def sanitize_for_ngrams(text: str) -> str:
    """Strip president names, years, hyphens, and pre-2009 diaeresis to avoid leakage."""
    t = LEAKAGE_PATTERN.sub(" ", text)
    # Normalize pre-2009 Portuguese orthographic diaeresis (ü -> u)
    t = t.replace("ü", "u").replace("Ü", "U")
    # Remove punctuation dashes and intra-word hyphens (stripped in Dilma corpus)
    t = t.replace("–", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_date_from_filename(path: Path) -> str:
    """Extract ISO-like date string from speech filename when present."""
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", path.stem)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"
    return path.parent.name


def main() -> None:
    records = []
    filter_stats = Counter()

    for folder_name, pres_name in PRESIDENT_MAP.items():
        pdir = RAW_BASE / folder_name
        if not pdir.exists():
            continue
        txt_files = sorted(
            [f for f in pdir.rglob("*.txt") if "Discursos" in f.parts or "Discurso" in str(f.parent)]
        )
        for tf in txt_files:
            filter_stats[f"raw_{pres_name}"] += 1
            raw_txt = decode_bytes(tf.read_bytes())
            clean_txt = clean_speech_text(raw_txt)

            if is_interview(clean_txt):
                filter_stats[f"drop_interview_{pres_name}"] += 1
                continue

            words = WORD_RE.findall(clean_txt.lower())
            wc = len(words)
            if wc < MIN_WORDS:
                filter_stats[f"drop_short_{pres_name}"] += 1
                continue

            if not is_portuguese(words):
                filter_stats[f"drop_non_pt_{pres_name}"] += 1
                continue

            records.append(
                {
                    "doc_id": f"{pres_name}_{tf.stem}_{tf.parent.name}",
                    "president": pres_name,
                    "date": parse_date_from_filename(tf),
                    "year": tf.parent.name if tf.parent.name.isdigit() else "",
                    "rel_path": str(tf.relative_to(PROJECT_ROOT)),
                    "word_count": wc,
                    "text_clean": clean_txt,
                }
            )

    df_all = pd.DataFrame(records)
    before_dedup = len(df_all)
    df_all = df_all.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)
    after_dedup = len(df_all)
    filter_stats["dedup_removed"] = before_dedup - after_dedup

    print("=== CLEANING & FILTERING SUMMARY ===")
    print(f"Total clean speeches before dedup: {before_dedup}, after dedup: {after_dedup}")
    print("\nClean counts by president (full dataset):")
    print(df_all["president"].value_counts().to_string())

    df_all.to_parquet(PROC_DIR / "discursos_all_clean.parquet", index=False)

    # Filter to presidents with substantial archive (>= 250 clean speeches) and balance sample
    df_eligible = df_all[df_all["president"].isin(PRESIDENT_ORDER)].copy()
    counts_eligible = df_eligible["president"].value_counts()
    n_per_pres = min(N_PER_PRESIDENT, int(counts_eligible.min()))
    print(f"\nBalanced sampling: {n_per_pres} speeches per president across {len(PRESIDENT_ORDER)} presidents.")

    balanced_parts = []
    for pres in PRESIDENT_ORDER:
        sub = df_eligible[df_eligible["president"] == pres]
        sampled = sub.sample(n=n_per_pres, random_state=SEED)
        balanced_parts.append(sampled)

    df_bal = pd.concat(balanced_parts, ignore_index=True)
    df_bal["president"] = pd.Categorical(df_bal["president"], categories=PRESIDENT_ORDER, ordered=True)
    df_bal = df_bal.sort_values(["president", "date", "doc_id"]).reset_index(drop=True)
    df_bal["president"] = df_bal["president"].astype(str)
    df_bal["text_no_leak"] = df_bal["text_clean"].map(sanitize_for_ngrams)

    print(f"Balanced corpus shape: {df_bal.shape}")
    print(df_bal.groupby("president")["word_count"].describe().round(1).to_string())

    # Extract Bloco A: 23 artifact-free interpretable stylometric features
    stylo_rows = [extract_stylo23(t) for t in df_bal["text_clean"]]
    df_stylo = pd.DataFrame(stylo_rows)
    stylo_cols = list(df_stylo.columns)
    assert not df_stylo.isna().any().any(), "NaN detected in stylometric features!"
    assert np.isfinite(df_stylo.to_numpy()).all(), "Inf detected in stylometric features!"

    scaler = StandardScaler()
    X_stylo23_z = scaler.fit_transform(df_stylo.to_numpy(dtype=np.float64))
    X_stylo23_l2 = normalize(X_stylo23_z, norm="l2")

    # Bloco B: Closed-class Portuguese function words (~200 words)
    func_vec = TfidfVectorizer(
        vocabulary=FUNCTION_WORDS,
        lowercase=True,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]+\b",
        sublinear_tf=True,
        norm="l2",
    )
    X_func = func_vec.fit_transform(df_bal["text_no_leak"]).toarray()
    # Keep only function words occurring in at least 5 documents
    keep_func = (X_func > 0).sum(axis=0) >= 5
    X_func = normalize(X_func[:, keep_func], norm="l2")
    func_names = [f"func_{w}" for w, k in zip(FUNCTION_WORDS, keep_func) if k]

    # Bloco C: Character 3-4-grams (morphological / syntactic style)
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=250,
        min_df=0.05,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )
    X_char = char_vec.fit_transform(df_bal["text_no_leak"]).toarray()
    char_names = [f"char_{g}" for g in char_vec.get_feature_names_out()]
    assert all("-" not in g for g in char_names), "Hyphen leaked into char_names!"

    # Combined high-dimensional pure-style matrix (equal L2 norm block weighting)
    X_style_highdim = normalize(np.hstack([X_stylo23_l2, X_func, X_char]), norm="l2")

    # Auxiliary lexical + style TF-IDF matrix (3000 word unigrams/bigrams) for comparison
    tfidf_vec = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]{2,}\b",
        norm="l2",
    )
    X_tfidf_3000 = tfidf_vec.fit_transform(df_bal["text_no_leak"])

    # Leakage verification
    leak_terms = {"sarney", "collor", "itamar", "fhc", "cardoso", "lula", "dilma", "rousseff", "temer", "bolsonaro", "jair"}
    leaked_func = [f for f in func_names if f.replace("func_", "") in leak_terms]
    assert not leaked_func, f"Leakage found in function words: {leaked_func}"

    print(f"\n=== FEATURE MATRICES ===")
    print(f"Bloco A (Stylo23):        {X_stylo23_z.shape}")
    print(f"Bloco B (Function words): {X_func.shape}")
    print(f"Bloco C (Char 3-4-grams): {X_char.shape}")
    print(f"Combined X_style_highdim: {X_style_highdim.shape}")
    print(f"Aux X_tfidf_3000:         {X_tfidf_3000.shape}")

    # Clean up old v1 filenames if present and persist updated artifacts
    (PROC_DIR / "features_stylo26.parquet").unlink(missing_ok=True)
    (PROC_DIR / "X_stylo26_z.npy").unlink(missing_ok=True)
    df_bal.to_parquet(PROC_DIR / "discursos_balanced.parquet", index=False)
    df_stylo.to_parquet(PROC_DIR / "features_stylo23.parquet", index=False)
    np.save(PROC_DIR / "X_stylo23_z.npy", X_stylo23_z)
    np.save(PROC_DIR / "X_style_highdim.npy", X_style_highdim)
    sparse.save_npz(PROC_DIR / "X_tfidf_3000.npz", X_tfidf_3000)

    meta = {
        "seed": SEED,
        "min_words": MIN_WORDS,
        "n_per_president": n_per_pres,
        "presidents": PRESIDENT_ORDER,
        "filter_stats": dict(filter_stats),
        "raw_counts_all_clean": df_all["president"].value_counts().to_dict(),
        "stylo_cols": stylo_cols,
        "func_names": func_names,
        "char_names": char_names,
        "dims": {
            "stylo23": X_stylo23_z.shape[1],
            "func": X_func.shape[1],
            "char": X_char.shape[1],
            "highdim_total": X_style_highdim.shape[1],
            "tfidf_3000": X_tfidf_3000.shape[1],
        },
    }
    (PROC_DIR / "feature_names.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved processed datasets and features to {PROC_DIR}")


if __name__ == "__main__":
    main()
