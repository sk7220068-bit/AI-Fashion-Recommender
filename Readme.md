# 🧥 AI Fashion Recommender & Outfit Upgrader

> An end-to-end AI-powered fashion intelligence system that detects clothing items from images, scores outfit compatibility, and suggests occasion-specific upgrades — all backed by computer vision and content-based machine learning.

---

## 🏗️ Architecture

```
User → React Frontend (port 5173)
         │
         ▼
   Spring Boot API (port 8080)
         │
   ┌─────┴─────────────────────────┐
   │  OutfitController             │
   │  RecommendationController     │
   │  UpgradeController            │
   └────────────────┬──────────────┘
                    │
        ┌───────────┼─────────────┐
        ▼           ▼             ▼
   MongoDB     Python ML      CSV Dataset
   (port 27017) Service       (outfits.csv)
               (port 5001)
                   │
              ┌────┴────┐
              │  YOLO   │  ResNet50
              └─────────┘
```

## 📦 Tech Stack

| Layer       | Technology                              |
|-------------|----------------------------------------|
| Backend     | Java 17, Spring Boot 3.2, Maven         |
| Database    | MongoDB 7.0                             |
| ML Service  | Python 3.11, Flask, YOLOv8, ResNet50   |
| Frontend    | React 18, Vite, Vanilla CSS             |
| Container   | Docker + docker-compose                 |

---

## 🚀 Quick Start

### Option A — Docker (Recommended, One Command)

```bash
# Clone and run everything
git clone <repo-url>
cd AI-Fashion-Recommender
docker-compose up --build
```

Open **http://localhost:5173** in your browser.

---

### Option B — Local Development (Three Terminals)

#### Prerequisites
- Java 17+ and Maven 3.9+
- Python 3.10+ and pip
- MongoDB running on `localhost:27017`
- Node.js 18+

#### Terminal 1 — MongoDB
```bash
# Start MongoDB (or use MongoDB Atlas)
mongod --dbpath /data/db
```

#### Terminal 2 — Python ML Service
```bash
cd ml-service
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5001
```

> **Note:** First run downloads YOLOv8 weights (~6 MB) and ResNet50 weights (~100 MB).
> Without a GPU, inference runs on CPU — slightly slower but fully functional.
> The app has **mock fallback** detection — it works even if the ML service is offline.

#### Terminal 3 — Spring Boot Backend
```bash
cd backend
mvn spring-boot:run
# Runs on http://localhost:8080
```

#### Terminal 4 — React Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:5173
```

---

## 🔌 REST API Reference

### `POST /api/upload-outfit`
Full pipeline: detect → features → compatibility → upgrade → recommendations.

```bash
curl -X POST http://localhost:8080/api/upload-outfit \
  -F "image=@outfit.jpg" \
  -F "occasion=party"
```

**Response:**
```json
{
  "occasion": "party",
  "detectedItems": [
    { "category": "t-shirt", "confidence": 0.93, "style": "casual", "formalityScore": 0.2 },
    { "category": "jeans",   "confidence": 0.88, "style": "casual", "formalityScore": 0.25 },
    { "category": "sneakers","confidence": 0.85, "style": "sporty", "formalityScore": 0.1 }
  ],
  "overallCompatibilityScore": 0.42,
  "itemsToReplace": ["sneakers", "t-shirt"],
  "upgradeSuggestions": [
    "Replace sneakers with ankle boots or heels for a party look",
    "Consider a silk blouse or fitted top instead of a plain T-shirt"
  ],
  "itemsToAdd": ["blazer", "fitted trousers", "clutch bag"],
  "upgradeSummary": "Your outfit scores 42% compatibility for party...",
  "recommendations": [ ... ]
}
```

---

### `POST /api/detect-clothes`
Clothing detection only (no upgrade logic).

```bash
curl -X POST http://localhost:8080/api/detect-clothes \
  -F "image=@outfit.jpg"
```

---

### `POST /api/recommend-outfit`
Get outfit recommendations by occasion/style (no image required).

```bash
curl -X POST http://localhost:8080/api/recommend-outfit \
  -H "Content-Type: application/json" \
  -d '{"occasion": "party", "style": "smart-casual"}'
```

---

### `POST /api/upgrade-outfit`
Get upgrade suggestions from pre-detected items (no image re-upload needed).

```bash
curl -X POST http://localhost:8080/api/upgrade-outfit \
  -H "Content-Type: application/json" \
  -d '{
    "detectedItems": [
      {"category": "t-shirt", "style": "casual", "formalityScore": 0.2},
      {"category": "jeans",   "style": "casual", "formalityScore": 0.25}
    ],
    "occasion": "party"
  }'
```

---

### `GET /api/upgrade-history`
Retrieve the 20 most recent upgrade analyses.

```bash
curl http://localhost:8080/api/upgrade-history
```

---

### `GET /api/health`
Health check.

```bash
curl http://localhost:8080/api/health
# {"status":"UP","service":"AI Fashion Recommender"}
```

---

## 🧠 How It Works

### Full Pipeline

```
1. User uploads outfit image + selects occasion
2. OutfitController receives the request
3. OutfitDetectionService → Python /detect endpoint
   └── YOLOv8 detects clothing items + bounding boxes
4. FeatureExtractionService → Python /extract-features endpoint
   └── ResNet50 extracts 2048-dim feature vectors per item
5. CompatibilityService + CompatibilityScorer
   └── Pairwise style, formality, and feature vector scoring
6. OutfitUpgradeEngine
   └── Occasion rules → itemsToReplace, itemsToAdd, suggestions
7. RecommendationService
   └── Cosine similarity against MongoDB outfit dataset
   └── Occasion + style boosting → ranked OutfitRecommendations
8. UpgradeResult saved to MongoDB (history)
9. JSON response → React frontend
```

### ML Fallback (Mock Mode)
All ML calls have graceful fallbacks — the app runs fully without the Python service:
- **Detection:** Returns a realistic T-shirt + jeans + sneakers mock outfit
- **Feature extraction:** Generates deterministic style-clustered feature vectors

---

## 📂 Project Structure

```
AI-Fashion-Recommender/
├── backend/                          # Java Spring Boot
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/com/fashionai/
│       ├── FashionAIApplication.java  # Entry point
│       ├── controller/               # REST endpoints
│       ├── service/                  # Business logic
│       ├── model/                    # Domain objects
│       ├── repository/               # MongoDB repos
│       ├── recommendation/           # Cosine similarity engine
│       ├── compatibility/            # Pair scoring
│       ├── upgrade/                  # Occasion rules engine
│       └── config/                   # CORS + MongoDB config
│
├── ml-service/                       # Python Flask ML
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py                        # Flask endpoints
│   ├── detection/clothing_detector.py # YOLOv8 wrapper
│   └── features/feature_extractor.py  # ResNet50 wrapper
│
├── frontend/                         # React + Vite
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                   # Main workflow
│       ├── components/               # UI components
│       └── styles/index.css          # Design system
│
├── dataset/
│   ├── outfits.csv                   # 25 sample outfits
│   └── deepfashion_sample.csv        # 100 DeepFashion rows
│
├── config/
│   └── model_config.yaml             # YOLO + ResNet config
│
├── docker-compose.yml
└── README.md
```

---

## 📊 Dataset

### `outfits.csv` (25 records)
Source dataset for recommendation engine. Schema:

| Column | Description |
|---|---|
| `id` | Unique outfit ID |
| `name` | Human-readable outfit name |
| `style` | casual / smart-casual / formal / sporty |
| `occasions` | Pipe-separated: `party\|work\|casual` |
| `items` | Pipe-separated clothing items |
| `color_palette` | Pipe-separated color names |
| `season` | spring / summer / autumn / winter / all |
| `formality_score` | 0.0 (very casual) → 1.0 (very formal) |
| `feature_vector` | 64-dim pipe-separated floats |

### `deepfashion_sample.csv` (100 records)
Schema-compatible subset of the DeepFashion dataset:
- 13 clothing categories (short_sleeve_top, trousers, dress, etc.)
- Attribute labels (fitted, slim, floral, etc.)
- Landmark coordinates
- Style and formality annotations

---

## ⚙️ Configuration

### `backend/src/main/resources/application.properties`

```properties
# MongoDB connection
spring.data.mongodb.uri=mongodb://localhost:27017/fashionai

# ML service URL
ml.service.base-url=http://localhost:5001

# File upload limit
spring.servlet.multipart.max-file-size=20MB

# Optional: OpenAI for generative descriptions
openai.api.enabled=false
openai.api.key=YOUR_KEY_HERE
```

### `config/model_config.yaml`
Tune YOLO confidence thresholds, feature dimensions, and occasion upgrade rules.

---

## 🤖 Optional: OpenAI Integration

To enable AI-generated outfit descriptions:

1. Set `openai.api.enabled=true` in `application.properties`
2. Set `openai.api.key=sk-...` (or env variable `OPENAI_API_KEY`)
3. The `OutfitUpgradeService` will call GPT-4o for natural language descriptions

---

## 🔧 Extending the Dataset (DeepFashion)

To use the full DeepFashion dataset:

1. Download from [DeepFashion](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
2. Run the seed script to import into MongoDB:

```bash
# Import outfits.csv into MongoDB
mongoimport --db fashionai --collection outfits \
  --type csv --headerline \
  --file dataset/outfits.csv
```

Or simply start the Spring Boot app — it auto-seeds on first run.

---

## 🧪 Running Tests

```bash
cd backend
mvn test
```

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Java Spring Boot · Python Flask · YOLOv8 · ResNet50 · React · MongoDB*
