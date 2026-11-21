"""Synthetic complaint generator -> data/complaints.csv and data/places.csv.

Run:  python -m ml.generate_data
Complaints are grouped into 'incidents' (cluster_id): several citizens report the same incident
from slightly different spots, in different languages. This gives ground truth for
duplicate-detection and hotspot-clustering evaluation.

IMPORTANT: this data is SYNTHETIC. Say so in the README and report metrics as 'on synthetic data'.
"""
import numpy as np
import pandas as pd

from app.config import CATEGORIES, DATA_DIR, INDUSTRIAL, LOCALITIES, OUTSKIRTS, SEVERITIES
from app.services.geo import nearest_locality, nearest_sensitive_m, offset_latlon

SEED = 42
N_INCIDENTS = 420

# ---------------- text lexicons ----------------
CAT_PHRASES = {
    "en": {
        "garbage_burning": ["garbage is being burned", "someone is burning waste", "open waste burning",
                            "trash fire", "burning of a garbage pile", "plastic and garbage being set on fire"],
        "construction_dust": ["construction site throwing dust everywhere", "heavy dust from building construction",
                              "construction work without dust covering", "debris and cement dust in the air",
                              "demolition dust", "dust from an under-construction building"],
        "industrial_smoke": ["factory chimney releasing black smoke", "industrial smoke from a unit",
                             "a factory is emitting toxic fumes", "smoke and chemical smell from industrial plant",
                             "factory burning something at night", "chimney emitting thick fumes"],
        "vehicle_emission": ["trucks emitting black smoke", "old diesel vehicles polluting",
                             "heavy vehicle traffic and fumes", "buses releasing thick exhaust",
                             "vehicles idling and emitting fumes", "smoky trucks parked with engines running"],
        "stubble_burning": ["farm residue is being burned", "crop stubble burning in nearby fields",
                            "fields on fire after harvest", "paddy straw being burnt",
                            "farmers burning crop waste", "stubble fire smoke drifting into the colony"],
    },
    "hinglish": {
        "garbage_burning": ["kachra jalaya ja raha hai", "koi kooda jala raha hai", "kachre ke dher mein aag lagi hai",
                            "plastic aur kachra jalaya ja raha hai", "khule mein kachra jal raha hai"],
        "construction_dust": ["construction site se bahut dhool uda rahi hai", "building ke kaam se dhool aur malba",
                              "bina dhake construction ki dhool hawa mein", "tod-phod se bahut dhool",
                              "cement ki dhool failti ja rahi hai"],
        "industrial_smoke": ["factory ki chimney se kaala dhuan nikal raha hai", "fakatri se zehrila dhuan",
                             "industry se chemical jaisi badbu aur dhuan", "raat ko factory mein kuch jalaya ja raha hai",
                             "chimney se gehra dhuan"],
        "vehicle_emission": ["trucks se kaala dhuan nikal raha hai", "purani diesel gaadiyan pradushan phaila rahi hain",
                             "jam mein gaadiyon ka dhuan", "bus se bahut dhuan nikal raha hai",
                             "trucks engine chalu karke khade hain aur dhuan"],
        "stubble_burning": ["khet mein parali jalayi ja rahi hai", "fasal ke avshesh jalaye ja rahe hain",
                            "khet mein aag aur dhuan colony tak aa raha hai", "kisan parali jala rahe hain",
                            "stubble jalane ka dhuan"],
    },
    "hi": {
        "garbage_burning": ["कचरा जलाया जा रहा है", "कोई कूड़ा जला रहा है", "कचरे के ढेर में आग लगी है",
                            "प्लास्टिक और कचरा जलाया जा रहा है", "खुले में कचरा जल रहा है"],
        "construction_dust": ["निर्माण स्थल से बहुत धूल उड़ रही है", "बिल्डिंग के काम से धूल और मलबा",
                              "बिना ढके निर्माण की धूल हवा में है", "तोड़फोड़ से बहुत धूल", "सीमेंट की धूल फैल रही है"],
        "industrial_smoke": ["फैक्ट्री की चिमनी से काला धुआं निकल रहा है", "फैक्ट्री से जहरीला धुआं",
                             "उद्योग से केमिकल जैसी बदबू और धुआं", "रात में फैक्ट्री में कुछ जलाया जा रहा है",
                             "चिमनी से गहरा धुआं"],
        "vehicle_emission": ["ट्रकों से काला धुआं निकल रहा है", "पुरानी डीजल गाड़ियां प्रदूषण फैला रही हैं",
                             "जाम में गाड़ियों का धुआं", "बस से बहुत धुआं निकल रहा है",
                             "ट्रक इंजन चालू करके खड़े हैं और धुआं"],
        "stubble_burning": ["खेत में पराली जलाई जा रही है", "फसल के अवशेष जलाए जा रहे हैं",
                            "खेत में आग और धुआं कॉलोनी तक आ रहा है", "किसान पराली जला रहे हैं", "पराली जलाने का धुआं"],
    },
}
SEV_SENTENCES = {
    "en": {
        "low": ["It is a small amount.", "Minor issue but please check.", "Not too much smoke yet."],
        "medium": ["It has been going on for a while.", "The smell is noticeable.", "Please look into it."],
        "high": ["Thick smoke is spreading in the area.", "Very heavy, hard to breathe.", "Residents are getting eye irritation."],
        "critical": ["Huge amount of smoke, children and elderly are coughing.", "Urgent! It is very dangerous.",
                     "Massive and out of control, people struggling to breathe."],
        "neutral": ["Please take action.", "Kindly check.", "Need this resolved."],
    },
    "hinglish": {
        "low": ["Thoda hi hai.", "Chhoti si problem hai par dekh lijiye.", "Abhi zyada dhuan nahi hai."],
        "medium": ["Kaafi der se chal raha hai.", "Badbu aa rahi hai.", "Please check kijiye."],
        "high": ["Bahut ghana dhuan fail raha hai.", "Saans lena mushkil ho raha hai.", "Aankhon mein jalan ho rahi hai."],
        "critical": ["Bahut zyada dhuan hai, bachche aur bujurg khaans rahe hain.", "Turant action chahiye, bahut khatarnak hai.",
                     "Aag kaabu se bahar hai, logon ko saans lene mein dikkat."],
        "neutral": ["Kripya action lijiye.", "Dekh lijiye.", "Ise theek karwaiye."],
    },
    "hi": {
        "low": ["थोड़ी सी ही है।", "छोटी समस्या है लेकिन देख लीजिए।", "अभी ज्यादा धुआं नहीं है।"],
        "medium": ["काफी देर से चल रहा है।", "बदबू आ रही है।", "कृपया जांच करें।"],
        "high": ["बहुत घना धुआं फैल रहा है।", "सांस लेना मुश्किल हो रहा है।", "आंखों में जलन हो रही है।"],
        "critical": ["बहुत ज्यादा धुआं है, बच्चे और बुजुर्ग खांस रहे हैं।", "तुरंत कार्रवाई चाहिए, बहुत खतरनाक है।",
                     "आग काबू से बाहर है, लोगों को सांस लेने में दिक्कत।"],
        "neutral": ["कृपया कार्रवाई करें।", "देख लीजिए।", "इसे ठीक करवाइए।"],
    },
}
VAGUE = {
    "en": ["the air is very bad", "so much smoke around", "pollution is getting worse"],
    "hinglish": ["yahan hawa bahut kharab hai", "bahut dhuan hai", "pradushan badh raha hai"],
    "hi": ["यहां हवा बहुत खराब है", "बहुत धुआं है", "प्रदूषण बढ़ रहा है"],
}
LOC_TEMPLATES = {"en": ["near {}", "in {}", "at {}"], "hinglish": ["{} ke paas", "{} mein"], "hi": ["{} के पास", "{} में"]}


def add_typos(s, rng):
    chars = list(s)
    idx = [i for i, c in enumerate(chars) if c.isascii() and c.isalpha()]
    for _ in range(int(rng.integers(1, 3))):
        if len(idx) < 3:
            break
        i = int(rng.choice(idx))
        if rng.random() < 0.5 and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars[i] = ""
    return "".join(chars)


def make_text(rng, lang, cat, sev_cue, locality):
    if rng.random() < 0.08:
        core = str(rng.choice(VAGUE[lang]))
    else:
        core = str(rng.choice(CAT_PHRASES[lang][cat]))
    loc = str(rng.choice(LOC_TEMPLATES[lang])).format(locality)
    sev_key = "neutral" if rng.random() < 0.30 else sev_cue
    sev_sentence = str(rng.choice(SEV_SENTENCES[lang][sev_key]))
    first = f"{core} {loc}."
    parts = [first, sev_sentence] if rng.random() < 0.8 else [sev_sentence, first]
    text = " ".join(parts)
    if lang != "hi":
        if rng.random() < 0.30:
            text = text.lower()
        else:
            text = text[0].upper() + text[1:]
        if rng.random() < 0.25:
            text = add_typos(text, rng)
    return text


def make_places(rng):
    rows = []
    for name, (lat, lon, _) in LOCALITIES.items():
        for kind, n in (("school", 3), ("hospital", 1)):
            for k in range(n):
                la, lo = offset_latlon(lat, lon, rng.uniform(-1500, 1500), rng.uniform(-1500, 1500))
                rows.append({"name": f"{name} {kind.title()} {k + 1} (synthetic)", "type": kind, "lat": la, "lon": lo})
    return pd.DataFrame(rows)


def shift(rng, sev, p):
    i = SEVERITIES.index(sev)
    if rng.random() < p:
        i = int(np.clip(i + rng.choice([-1, 1]), 0, len(SEVERITIES) - 1))
    return SEVERITIES[i]


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(exist_ok=True)
    places = make_places(rng)
    places.to_csv(DATA_DIR / "places.csv", index=False)

    all_loc = list(LOCALITIES)
    rows, cid_counter = [], 0
    p_near = {"low": 0.10, "medium": 0.20, "high": 0.40, "critical": 0.70}

    for cluster_id in range(N_INCIDENTS):
        sev = str(rng.choice(SEVERITIES, p=[0.25, 0.35, 0.28, 0.12]))
        cat = str(rng.choice(CATEGORIES, p=[0.30, 0.25, 0.15, 0.20, 0.10]))
        pool = OUTSKIRTS if cat == "stubble_burning" else INDUSTRIAL if cat == "industrial_smoke" else all_loc
        if rng.random() < p_near[sev]:
            pl = places.iloc[int(rng.integers(len(places)))]
            clat, clon = offset_latlon(pl.lat, pl.lon, rng.uniform(-350, 350), rng.uniform(-350, 350))
            locality, _ = nearest_locality(clat, clon)
        else:
            locality = str(rng.choice(pool))
            la, lo, _ = LOCALITIES[locality]
            clat, clon = offset_latlon(la, lo, rng.uniform(-1500, 1500), rng.uniform(-1500, 1500))
        base_hours = float(np.clip(rng.lognormal(1.8 + 0.25 * SEVERITIES.index(sev), 0.8), 1, 96))
        size = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 6, 10]))

        for _ in range(size):
            lat, lon = offset_latlon(clat, clon, rng.normal(0, 30), rng.normal(0, 30))
            label_sev = shift(rng, sev, 0.12)
            cue_sev = shift(rng, label_sev, 0.08)
            lang = str(rng.choice(["en", "hinglish", "hi"], p=[0.45, 0.35, 0.20]))
            rows.append({
                "complaint_id": cid_counter, "cluster_id": cluster_id,
                "text": make_text(rng, lang, cat, cue_sev, locality), "language": lang,
                "category": cat, "severity": label_sev, "lat": round(lat, 6), "lon": round(lon, 6),
                "locality": locality, "nearest_sensitive_m": round(nearest_sensitive_m(lat, lon, places), 1),
                "duration_hours": round(base_hours * rng.uniform(0.5, 1.0), 1),
            })
            cid_counter += 1

    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "complaints.csv", index=False)
    print(f"Wrote {len(df)} complaints in {df.cluster_id.nunique()} incidents, {len(places)} places")
    print(df.category.value_counts().to_string()); print(df.severity.value_counts().to_string())
    print(df.language.value_counts().to_string())


if __name__ == "__main__":
    main()
