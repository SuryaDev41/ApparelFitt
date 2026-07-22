# StyleSense AI — Project Explanation

## Overview

StyleSense AI is a Python terminal chatbot that provides fashion and apparel guidance only. It is designed to help a user choose outfits, understand proportion-based body shapes, plan a wardrobe, select colours and fabrics, and dress for occasions.

The application can run in three complementary modes:

- **Knowledge-base (RAG) mode:** Common fashion questions are answered directly from a curated knowledge base retrieved via a Qdrant vector search — **without any LLM call**.
- **Grok mode:** Novel questions are sent to the xAI Grok API, grounded with any relevant retrieved context.
- **Local demo mode:** Gives a small set of built-in fashion replies when no API key is configured or the API is unavailable.

The RAG layer is optional and degrades gracefully: if Qdrant is not configured or `qdrant-client` is not installed, the app runs exactly as a Grok + local-rules chatbot.

## Technology

| Area | Choice |
| --- | --- |
| Language | Python 3.9+ |
| Interface | Terminal / PowerShell |
| AI provider | xAI Grok Chat Completions API |
| Vector store | Qdrant Cloud (optional) |
| Embeddings | fastembed `BAAI/bge-small-en-v1.5`, computed locally |
| External packages | `qdrant-client[fastembed]` (only for the optional RAG layer) |
| Configuration | `.env` file read by the application |

The core chatbot uses Python standard-library modules only (`json`, `urllib`, `re`, `os`, `pathlib`). The optional retrieval layer adds `qdrant-client[fastembed]`.

## Main Files

| File | Purpose |
| --- | --- |
| `stylesense.py` | The complete terminal chatbot application. |
| `rag.py` | Optional RAG layer: local fastembed embeddings + Qdrant Cloud retrieval. |
| `knowledge_base.json` | Curated fashion knowledge that is embedded and searched. |
| `ingest.py` | One-off script to build/rebuild the Qdrant collection from the knowledge base. |
| `requirements.txt` | Python dependencies for the RAG layer. |
| `.env` | Your private local configuration (API keys, Qdrant creds). Not committed. |
| `.env.example` | A safe template for the `.env` file. |
| `.gitignore` | Keeps keys and Python cache files out of Git. |

## Starting the Application

Run the following command from the project folder:

```powershell
python .\stylesense.py
```

At startup, the chatbot introduces itself and shows a guided menu:

```text
1. Find my body shape and best fits
2. Style an occasion
3. Build a wardrobe or capsule
4. Ask another fashion question
```

The user should enter the number of the desired option. The application then asks the follow-up questions needed for that path.

## Conversation Paths

### 1. Body-shape and best-fit analysis

The app asks for:

- Male or female measurement guide
- Height (optional)
- Shoulder width
- Chest or bust circumference
- Waist circumference
- Hip circumference

It validates that the required measurements are positive numbers. It does **not** ask for or use body weight.

The body-shape result is calculated from upper-body, waist, and hip proportions. The output includes a positive visual-balance explanation, fit direction, fabric and colour-placement guidance, a styling tip, and a common mistake to avoid.

### 2. Occasion styling

The app asks for the occasion, climate/season, budget, and preferred style direction. It turns those details into a fashion prompt for Grok or the local fallback.

### 3. Wardrobe or capsule planning

The app asks for the user’s climate, budget, and preferred style direction, then requests a capsule-wardrobe recommendation.

### 4. Open fashion question

The user can ask about clothing, fit, colours, fabrics, footwear, accessories, or styling.

## Fashion-Only Safety Gate

Before a message is sent to Grok, the application checks it for fashion-related terms. Unrelated questions are blocked locally and receive this exact response:

```text
I'm your dedicated AI Fashion Stylist. I can help you with clothing, body shape analysis, styling advice, outfits, fashion trends, colors, fabrics, and accessories. Feel free to ask me anything related to fashion!
```

This protects the intended scope even if an API key is configured.

## Grok API Configuration

1. Create the local configuration file:

```powershell
Copy-Item .env.example .env
```

2. Open `.env` and add your key:

```text
XAI_API_KEY=your-real-xai-api-key
```

3. Run the application again.

Optional `.env` values:

```text
XAI_MODEL=grok-3-latest
XAI_API_URL=https://api.x.ai/v1/chat/completions
```

The program loads `.env` automatically when it starts. Environment variables already set in the operating system take priority over the values in `.env`.

## Retrieval-Augmented Generation (RAG)

The RAG layer reduces LLM calls by answering common questions from a local knowledge base.

**How it works**

1. `knowledge_base.json` holds curated fashion answers (body shapes, colour theory, fit, occasions, fabrics, care, accessories, aesthetics, proportion).
2. `ingest.py` embeds each entry locally with fastembed (no embedding-API calls) and upserts the vectors into a Qdrant Cloud collection. Ingestion is idempotent and also runs automatically on first launch if the collection is empty.
3. On each fashion question the app embeds the query locally and searches Qdrant for the closest entries.

**Decision thresholds** (cosine similarity, configurable in `.env`):

| Condition | Action | LLM call? |
| --- | --- | --- |
| top score ≥ `RAG_ANSWER_THRESHOLD` (default 0.85) | Answer straight from the knowledge base | **No** |
| top score ≥ `RAG_CONTEXT_THRESHOLD` (default 0.60) | Call Grok, grounded with retrieved notes | Yes (better grounded) |
| below both | Call Grok normally (or local rules offline) | Yes |

This means the most-asked questions are served instantly and for free, while Grok is reserved for genuinely novel queries.

### Setup

```powershell
pip install -r requirements.txt
# Add QDRANT_URL and QDRANT_API_KEY to .env (see .env.example)
python ingest.py            # build the collection (use --force to rebuild)
python .\stylesense.py
```

To grow the assistant's free/instant coverage, add entries to `knowledge_base.json` and re-run `python ingest.py --force`.

## Internal Flow

```text
User input
    ↓
Guided menu or chat command
    ↓
Fashion-intent validation
    ├── Non-fashion → predefined refusal response
    └── Fashion → body-shape flow
                  or Qdrant retrieval
                       ├── strong match → knowledge-base answer (no LLM call)
                       ├── partial match → Grok grounded with retrieved notes
                       └── no match → Grok (or local advice offline)
                              ↓
                     Styling response in terminal
```

## Commands

| Command | Action |
| --- | --- |
| `/profile` | Save optional style and measurement information. |
| `/reset` | Clear the current conversation history. |
| `/help` | Display available commands. |
| `/exit` or `/quit` | Close the program. |

## Error Handling

- Invalid measurements are rejected and requested again.
- If the Grok API is unreachable, returns an error, or sends invalid JSON, the app falls back to local fashion guidance.
- Pressing `Ctrl+C` or ending terminal input exits cleanly.

## Future Improvements

- Save profiles and conversation history to a local database.
- Add a richer local recommendation engine for offline use.
- Improve natural-language menu selection.
- Add images or a web interface later without changing the core styling logic.
- Add unit tests for the body-shape classification thresholds and API error handling.
