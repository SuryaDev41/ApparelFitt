#!/usr/bin/env python3
"""StyleSense AI: terminal-first, fashion-only personal stylist."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import rag  # optional RAG layer; degrades gracefully when Qdrant is not configured


def load_env_file() -> None:
    """Load simple KEY=VALUE pairs from .env beside this script.

    Existing system environment variables take precedence, so deployment settings
    can override local values. No external dotenv package is needed.
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()

FASHION_REFUSAL = (
    "I'm your dedicated AI Fashion Stylist. I can help you with clothing, "
    "body shape analysis, styling advice, outfits, fashion trends, colors, "
    "fabrics, and accessories. Feel free to ask me anything related to fashion!"
)
API_URL = os.getenv("XAI_API_URL", "https://api.x.ai/v1/chat/completions")
MODEL = os.getenv("XAI_MODEL", "grok-3-latest")
API_KEY = os.getenv("XAI_API_KEY")

# RAG retrieval thresholds (cosine similarity, 0..1). At or above the answer
# threshold a question is served directly from the knowledge base with no LLM call;
# at or above the context threshold Grok is still called but grounded with retrieved
# notes. Tune per embedding model via .env.
RAG_ANSWER_THRESHOLD = float(os.getenv("RAG_ANSWER_THRESHOLD", "0.85"))
RAG_CONTEXT_THRESHOLD = float(os.getenv("RAG_CONTEXT_THRESHOLD", "0.60"))
SHOW_SOURCE = os.getenv("STYLESENSE_SHOW_SOURCE", "1").strip().lower() not in ("0", "false", "no")

SYSTEM_PROMPT = f"""You are StyleSense AI, a specialized AI Fashion Stylist.
Your expertise covers clothing, fashion, apparel, personal styling, outfit recommendations, body-shape analysis, trends, colour coordination, fabrics, accessories, seasonal fashion, wardrobes, footwear, grooming related to fashion, occasion dressing, brands and styling tips. General fashion-knowledge questions (e.g. "what is quiet luxury?", "how should a blazer fit?") are in scope.

Only decline when a request is clearly unrelated to fashion or apparel. In that case respond with exactly this text and nothing else: {FASHION_REFUSAL}

Answer directly and completely in a warm, natural, conversational tone. Give the actual recommendation first, then the reasoning. Only ask a clarifying question when you genuinely cannot give a useful answer without it — never open with a list of questions. If some details are missing, make sensible assumptions, state them briefly, and still give a concrete answer. Treat short or referential messages ("why?", "what about winter?", "and shoes?") as follow-ups to the ongoing styling conversation.

For body-shape analysis, use proportions, never body weight. Before determining a body shape, request: male—shoulder, chest, waist, hip; female—shoulder, bust, waist, hip. Do not guess missing measurements.

Recommendations should consider body shape, height, skin tone when available, personal style, occasion, budget, climate and trends. Explain why the pieces suit proportions, visual balance, fit, colours, fabrics, styling details and common mistakes. Never body shame or rank body shapes. Use concise Markdown headings and bullets, and keep answers focused rather than exhaustive."""

FASHION_TERMS = re.compile(
    r"\b("
    r"cloth|apparel|garment|fashion|styl|outfit|ensemble|look|lookbook|"
    r"wardrobe|capsule|closet|"
    r"dress|gown|shirt|t[ -]?shirt|tee|top|blouse|polo|jean|denim|chino|khaki|"
    r"trouser|pant|slack|short|skirt|legging|jogger|cargo|"
    r"suit|tuxedo|blazer|jacket|coat|overcoat|parka|trench|"
    r"sweater|jumper|knit|cardigan|hoodie|sweatshirt|vest|waistcoat|"
    r"wear|dressing|"
    r"shoe|boot|loafer|sneaker|trainer|heel|pump|sandal|flat|oxford|brogue|derby|"
    r"footwear|sock|"
    r"bag|handbag|clutch|backpack|belt|watch|tie|scarf|glove|hat|cap|beanie|"
    r"sunglass|cufflink|jewell?er|necklace|bracelet|earring|accessor|"
    r"fabric|material|textile|linen|cotton|wool|silk|leather|suede|tweed|"
    r"corduroy|cashmere|chiffon|satin|velvet|canvas|flannel|jersey|polyester|"
    r"viscose|rayon|nylon|spandex|"
    r"fit|tailor|hem|cuff|sleeve|collar|neckline|waistline|inseam|drape|silhouette|"
    r"colour|color|palette|undertone|neutral|pastel|contrast|monochrome|"
    r"pattern|print|stripe|plaid|check|floral|"
    r"skin tone|body shape|body type|proportion|figure|petite|curvy|plus[ -]?size|athletic|"
    r"size|sizing|measurement|"
    r"grooming|haircut|hairstyle|beard|"
    r"occasion|formal|business|casual|smart casual|cocktail|black[ -]?tie|"
    r"wedding|interview|brunch|party|gala|festival|vacation|office|gym|"
    r"season|summer|winter|autumn|spring|monsoon|"
    r"streetwear|athleisure|minimalist|boho|preppy|grunge|vintage|retro|chic|elegant|"
    r"trend|aesthetic|vibe|quiet luxury|old money|"
    r"ethnic|saree|sari|kurta|kurti|lehenga|sherwani|salwar|"
    r"matching|pair with|coordinate|combo|combination|goes with|go with|wear with|layer|"
    r"flatter|slim|taller|elongate)\b",
    re.IGNORECASE,
)
BODY_SHAPE_TERMS = re.compile(r"\b(body shape|body type|figure|silhouette|measurements?|proportions?)\b", re.IGNORECASE)
OFF_TOPIC_TERMS = re.compile(
    r"\b(cricket|football|soccer|basketball|tennis|hockey|tournament|"
    r"politic|election|president|minister|government|"
    r"stock|crypto|bitcoin|invest|"
    r"weather forecast|"
    r"recipe|cook|bake|"
    r"python|javascript|programming|algorithm|software|"
    r"quantum|physics|chemistry|biology|calculus|equation|"
    r"capital of|"
    r"disease|symptom|diagnos|"
    r"lyrics)\b",
    re.IGNORECASE,
)
FOLLOWUP_HINT = re.compile(
    r"\b(it|its|that|this|them|those|these|they|instead|also|too|same|"
    r"what about|how about|why|which|when|where|and|or|more|else|another|other|"
    r"different|alternative|version|option|prefer|rather|yes|no|sure|okay|ok|thanks)\b",
    re.IGNORECASE,
)


def is_fashion_query(message: str) -> bool:
    """Allow-list of fashion vocabulary so obvious off-topic can be short-circuited."""
    return bool(FASHION_TERMS.search(message))


def is_followup(message: str, history: list[dict[str, str]]) -> bool:
    """In an ongoing styling chat, short or referential replies count as follow-ups.

    This keeps natural conversation flowing ("why?", "what about winter?", "and shoes?")
    without demanding a fashion keyword in every message. The model's own system prompt
    still declines anything genuinely off-topic that slips through.
    """
    if not history:
        return False
    if len(message.split()) <= 14:
        return True
    return bool(FOLLOWUP_HINT.search(message))


def is_on_topic(message: str, history: list[dict[str, str]]) -> bool:
    """Fashion keywords always pass; obvious off-topic never does; otherwise allow
    conversational follow-ups so natural dialogue is not interrupted."""
    if is_fashion_query(message):
        return True
    if OFF_TOPIC_TERMS.search(message):
        return False
    return is_followup(message, history)


def local_reply(message: str) -> str:
    """Useful fashion fallback when no Grok API key is configured or the API fails."""
    query = message.lower()
    if re.search(r"body shape|body type|measurement|proportion|figure", query):
        return (
            "To assess body shape from proportions, please share your shoulder, "
            "chest/bust, waist and hip measurements, plus height. I will not use body weight."
        )
    if re.search(r"color|colour|skin tone|undertone|palette", query):
        return (
            "For colour guidance, tell me your skin-tone undertone (warm, cool or neutral), "
            "hair colour and the setting. I can then recommend clothing colours, contrast "
            "level and accessory metals."
        )
    if re.search(r"capsule|wardrobe|closet", query):
        return (
            "A versatile capsule starts with well-fitting elevated basics: quality tees or "
            "shirts, a tailored layer, dark straight-leg denim, tailored trousers, a knit, "
            "and one refined pair each of casual and dress shoes. Tell me your climate, "
            "budget and preferred style so I can tailor the palette and fabric choices."
        )
    if re.search(r"wedding|interview|party|cocktail|black[ -]?tie|formal|occasion|gala|date", query):
        return (
            "For occasion dressing, a safe, sharp default is a well-fitted mid-to-dark outfit "
            "with one refined layer (blazer or structured knit) and clean leather footwear. "
            "Tell me the exact occasion, dress code, climate and your budget, and I'll tailor "
            "the silhouette, colours and fabrics."
        )
    if re.search(r"shoe|boot|sneaker|loafer|heel|footwear|sandal", query):
        return (
            "Footwear should match the formality and colour temperature of the outfit: "
            "leather loafers or derbies dress a look up, clean minimal sneakers keep it casual. "
            "Tell me the outfit and occasion and I'll suggest a style, colour and material."
        )
    if re.search(r"fabric|material|linen|cotton|wool|silk|leather|cashmere", query):
        return (
            "Fabric choice depends on climate and formality: breathable cotton and linen for "
            "heat, wool and knits for cold, and structured weaves for formal wear. Tell me the "
            "setting and I'll recommend fabrics that drape and hold their line well."
        )
    return (
        "Tell me the occasion, climate, budget, height, preferred style and any item you "
        "want to wear. I'll build an outfit with the right silhouette, fit, colours, fabrics "
        "and finishing details."
    )


def is_body_shape_request(message: str) -> bool:
    return bool(BODY_SHAPE_TERMS.search(message))


def ask_positive_number(prompt: str) -> float:
    """Request a usable measurement without guessing missing values."""
    while True:
        value = input(prompt).strip()
        try:
            number = float(value)
            if number > 0:
                return number
        except ValueError:
            pass
        print("Please enter a positive number in centimetres (for example, 82).")


def analyse_body_shape(profile: dict[str, str], guide: str) -> str:
    """Classify proportions from the four required measurements, never weight."""
    chest_or_bust = float(profile["chest/bust"])
    waist = float(profile["waist"])
    hip = float(profile["hip"])
    shoulder = float(profile["shoulder"])
    upper_lower_gap = (chest_or_bust - hip) / max(chest_or_bust, hip)
    waist_definition = 1 - waist / ((chest_or_bust + hip) / 2)

    if waist >= min(chest_or_bust, hip) * 0.95:
        shape = "oval / apple-balanced"
        balance = "A clean, uninterrupted vertical line will create elegant visual length."
        fit = "Choose softly structured layers, open necklines and straight or gently tapered trousers/skirts."
    elif guide == "female" and abs(upper_lower_gap) <= 0.05 and waist_definition >= 0.20:
        shape = "hourglass"
        balance = "Your bust and hip proportions are balanced with a defined waist."
        fit = "Choose pieces that follow the waist without pulling: wrap tops, shaped blazers and high-rise bottoms work especially well."
    elif upper_lower_gap <= -0.06:
        shape = "triangle / pear"
        balance = "Your lower proportion is more prominent than your upper proportion, creating a grounded silhouette."
        fit = "Use softly structured shoulders, brighter or textured tops, and clean darker bottoms to balance the eye."
    elif upper_lower_gap >= 0.06:
        shape = "inverted triangle"
        balance = "Your upper proportion is more prominent, which creates a naturally strong shoulder line."
        fit = "Choose relaxed shoulder construction, open necklines and straight or gently fuller lower layers for visual balance."
    elif waist_definition < 0.14:
        shape = "rectangle / straight"
        balance = "Your upper and lower proportions are close, with a subtle waist transition."
        fit = "Create definition with tailored layers, half-tucks, belts or high-rise bottoms while keeping the silhouette easy."
    else:
        shape = "balanced trapezoid"
        balance = "Your proportions transition gradually from upper body to waist and hip."
        fit = "Well-fitted shirts, lightly structured jackets and straight or tapered bottoms will preserve that balance."

    shoulder_note = (
        "Your shoulder measurement is recorded as a fit reference: ensure shoulder seams end at your natural shoulder point."
        if shoulder else ""
    )
    label = "Bust" if guide == "female" else "Chest"
    return (
        f"\nYour proportion-based style direction: **{shape.title()}**\n\n"
        f"- **Why:** {balance}\n"
        f"- **Fit:** {fit}\n"
        f"- **Colour & fabric:** Use medium-weight fabrics with enough structure to hold the desired line; place lighter colours or texture where you want visual emphasis.\n"
        f"- **Styling tip:** Keep the {label.lower()}–waist–hip fit comfortable and smooth rather than tight.\n"
        f"- **Avoid:** Sizing down for a closer fit; it interrupts the silhouette instead of defining it.\n"
        f"- **Measurement note:** {shoulder_note}"
    )


def ask_body_shape_measurements(profile: dict[str, str]) -> str:
    print("\nI’ll use proportions only — never body weight.")
    while True:
        guide = input("Use which measurement guide? [male/female]: ").strip().lower()
        if guide in ("male", "female"):
            break
        print("Please enter 'male' or 'female' so I can label the chest/bust measurement correctly.")

    profile["height"] = input("Height (e.g. 178 cm, optional): ").strip()
    profile["shoulder"] = str(ask_positive_number("Shoulder width in cm (across the back): "))
    label = "Bust" if guide == "female" else "Chest"
    profile["chest/bust"] = str(ask_positive_number(f"{label} circumference in cm: "))
    profile["waist"] = str(ask_positive_number("Waist circumference in cm: "))
    profile["hip"] = str(ask_positive_number("Hip circumference in cm: "))
    return analyse_body_shape(profile, guide)


def profile_summary(profile: dict[str, str]) -> str:
    details = [f"{key}: {value}" for key, value in profile.items() if value]
    return f"\nStyling profile: {', '.join(details)}." if details else ""


def ask_profile(profile: dict[str, str]) -> None:
    print("\nStyleSense profile — leave any field blank to skip.")
    print("Measurements are used only for proportion-based advice, never weight.")
    prompts = (
        ("height", "Height (e.g. 178 cm): "),
        ("skin tone", "Skin tone / undertone (optional): "),
        ("style", "Preferred style (optional): "),
        ("shoulder", "Shoulder measurement (cm): "),
        ("chest/bust", "Chest or bust measurement (cm): "),
        ("waist", "Waist measurement (cm): "),
        ("hip", "Hip measurement (cm): "),
    )
    for key, prompt in prompts:
        profile[key] = input(prompt).strip()
    print("\nProfile saved.")


def grok_reply(message: str, history: list[dict[str, str]], profile: dict[str, str], context: str = "") -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": message + profile_summary(profile) + context},
        ],
        "temperature": 0.65,
        "max_tokens": 1200,
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or "I could not generate a styling response. Please try again."


def format_kb_answer(hit: dict) -> str:
    """Return a knowledge-base answer, optionally marked as served without an LLM call."""
    answer = hit["answer"]
    if SHOW_SOURCE:
        answer += "\n\n_· from StyleSense knowledge base — no API call_"
    return answer


def build_context(hits: list[dict]) -> str:
    """Condense retrieved notes into a background block for grounding Grok."""
    notes = []
    for hit in hits[:3]:
        snippet = re.sub(r"\s+", " ", hit["answer"]).strip()
        notes.append(f"- {snippet[:400]}")
    return (
        "\n\n[Reference notes from the StyleSense knowledge base — use if relevant, "
        "do not mention or quote them]\n" + "\n".join(notes)
    )


def respond(message: str, history: list[dict[str, str]], profile: dict[str, str]) -> str:
    """Gate to fashion scope, then answer via RAG-first retrieval, Grok, or local rules.

    Order of preference (each step avoids an LLM call where possible):
      1. Strong knowledge-base match  -> serve it directly (no LLM call).
      2. No Grok key                  -> a KB match if decent, else local rules.
      3. Otherwise                    -> Grok, grounded with retrieved notes when relevant.
    """
    if not is_on_topic(message, history):
        return FASHION_REFUSAL

    hits = rag.retrieve(message) if rag.rag_enabled() else []
    top = hits[0] if hits else None

    if top and top["score"] >= RAG_ANSWER_THRESHOLD:
        return format_kb_answer(top)

    if not API_KEY:
        if top and top["score"] >= RAG_CONTEXT_THRESHOLD:
            return format_kb_answer(top)
        return local_reply(message)

    context = build_context(hits) if (top and top["score"] >= RAG_CONTEXT_THRESHOLD) else ""
    try:
        return grok_reply(message, history, profile, context)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
        print(f"\nGrok connection failed. Using local guidance instead. ({error})")
        if top and top["score"] >= RAG_CONTEXT_THRESHOLD:
            return format_kb_answer(top)
        return local_reply(message)


def save_turn(history: list[dict[str, str]], message: str, answer: str) -> None:
    history.extend(({"role": "user", "content": message}, {"role": "assistant", "content": answer}))
    if len(history) > 12:
        del history[:-12]


def interpret_guided_choice(value: str) -> str:
    """Accept both menu digits and the natural language users commonly type."""
    choice = value.strip().lower()
    if choice[:1] in {"1", "2", "3", "4"}:
        return choice[:1]
    if is_body_shape_request(choice) or re.search(r"\bbest fit|find my fit\b", choice):
        return "1"
    if re.search(r"\b(occasion|event|wedding|dinner|party|interview|workwear)\b", choice):
        return "2"
    if re.search(r"\b(wardrobe|capsule|closet)\b", choice):
        return "3"
    if re.search(r"\b(ask|question|help)\b", choice):
        return "4"
    return ""


def run_guided_choice(choice: str, history: list[dict[str, str]], profile: dict[str, str]) -> bool:
    """Run a selected onboarding branch. Returns False for an unrecognised choice."""
    if choice == "1":
        answer = ask_body_shape_measurements(profile)
        print(f"\nStyleSense › {answer}\n")
    elif choice == "2":
        occasion = input("What is the occasion? ").strip()
        climate = input("What climate or season will it be? ").strip()
        budget = input("What is your budget range? ").strip()
        style = input("What style should the look lean toward? ").strip()
        message = f"Style an outfit for {occasion or 'an occasion'}, climate: {climate or 'not specified'}, budget: {budget or 'not specified'}, preferred style: {style or 'not specified'}."
        answer = respond(message, history, profile)
        print(f"\nStyleSense › {answer}\n")
        save_turn(history, message, answer)
    elif choice == "3":
        climate = input("What climate do you dress for most? ").strip()
        budget = input("What is your budget range? ").strip()
        style = input("Which style direction do you enjoy? ").strip()
        message = f"Build a capsule wardrobe for climate: {climate or 'not specified'}, budget: {budget or 'not specified'}, preferred style: {style or 'not specified'}."
        answer = respond(message, history, profile)
        print(f"\nStyleSense › {answer}\n")
        save_turn(history, message, answer)
    elif choice == "4":
        print("\nGreat — ask me anything about clothing, fit, colours, fabrics or accessories.\n")
    else:
        return False
    return True


def start_guided_conversation(history: list[dict[str, str]], profile: dict[str, str]) -> None:
    """Open with a decision tree so users are guided from their first prompt."""
    print("\nHi — I’m StyleSense, your personal fashion stylist.")
    print("I’ll help you find clothing that suits your proportions, style and real life.")
    print("\nWhat would you like to start with?")
    print("  1. Find my body shape and best fits")
    print("  2. Style an occasion")
    print("  3. Build a wardrobe or capsule")
    print("  4. Ask another fashion question")
    raw_choice = input("Choose 1–4, or describe what you need: ")
    choice = interpret_guided_choice(raw_choice)
    if choice:
        run_guided_choice(choice, history, profile)
    elif raw_choice.strip():
        # Treat an unrecognised sentence as a normal first chat message.
        answer = respond(raw_choice, history, profile)
        print(f"\nStyleSense › {answer}\n")
        save_turn(history, raw_choice, answer)


def main() -> None:
    history: list[dict[str, str]] = []
    profile: dict[str, str] = {}
    print("\nStyleSense AI — your personal fashion stylist")
    print(f"Grok mode enabled ({MODEL})." if API_KEY else "Local demo mode enabled. Add XAI_API_KEY to use Grok.")
    if rag.rag_enabled():
        print("Preparing knowledge base (Qdrant)…", end=" ", flush=True)
        print("ready — common questions answered without an API call." if rag.ensure_ready() else "unavailable, continuing without it.")
    print("Type /help for commands.\n")
    start_guided_conversation(history, profile)

    while True:
        try:
            message = input("You › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message in ("/exit", "/quit"):
            break
        if message == "/help":
            print("\n/profile  update style profile\n/reset    clear conversation history\n/exit     quit\n")
            continue
        if message == "/profile":
            ask_profile(profile)
            continue
        if message == "/reset":
            history.clear()
            print("\nNew styling conversation started.\n")
            continue

        if message in {"1", "2", "3", "4"}:
            run_guided_choice(message, history, profile)
            continue

        if is_body_shape_request(message):
            answer = ask_body_shape_measurements(profile)
        else:
            answer = respond(message, history, profile)

        print(f"\nStyleSense › {answer}\n")
        save_turn(history, message, answer)

    print("\nStay well dressed. Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nStay well dressed. Goodbye!")
        sys.exit(0)
