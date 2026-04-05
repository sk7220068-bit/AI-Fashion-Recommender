import { useState } from 'react'
import axios from 'axios'
import ImageUploader from './components/ImageUploader'
import OccasionSelector from './components/OccasionSelector'
import DetectionResults from './components/DetectionResults'
import UpgradePanel from './components/UpgradePanel'

// ── Steps definition ──────────────────────────────────────────────
const STEPS = [
  { id: 'upload',    label: 'Upload'    },
  { id: 'detect',   label: 'Detect'    },
  { id: 'upgrade',  label: 'Upgrade'   },
  { id: 'results',  label: 'Results'   },
]

// ── Loading messages shown during API processing ──────────────────
const LOADING_STEPS = [
  { text: 'Analysing image...',           delay: 0    },
  { text: 'Detecting clothing items...',  delay: 1200 },
  { text: 'Extracting visual features...',delay: 2600 },
  { text: 'Scoring compatibility...',     delay: 4000 },
  { text: 'Generating upgrade plan...',   delay: 5400 },
  { text: 'Finding recommendations...',  delay: 6800 },
]

export default function App() {
  // ── State ──────────────────────────────────────────────────────
  const [imageFile,     setImageFile]     = useState(null)
  const [previewUrl,    setPreviewUrl]    = useState(null)
  const [occasion,      setOccasion]      = useState('casual')
  const [currentStep,   setCurrentStep]   = useState(0)
  const [loading,       setLoading]       = useState(false)
  const [loadingStep,   setLoadingStep]   = useState(0)
  const [error,         setError]         = useState(null)
  const [detectedItems, setDetectedItems] = useState([])
  const [upgradeResult, setUpgradeResult] = useState(null)

  // ── Handlers ───────────────────────────────────────────────────
  const handleImageSelect = (file, url) => {
    setImageFile(file)
    setPreviewUrl(url)
    setDetectedItems([])
    setUpgradeResult(null)
    setError(null)
  }

  const handleAnalyse = async () => {
    if (!imageFile) { setError('Please upload an outfit image first.'); return }
    if (!occasion)  { setError('Please select an occasion.'); return }

    setError(null)
    setLoading(true)
    setCurrentStep(1)
    setLoadingStep(0)

    // Advance loading step indicators with delays
    LOADING_STEPS.forEach((step, i) => {
      setTimeout(() => setLoadingStep(i), step.delay)
    })

    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      formData.append('occasion', occasion)

      const { data } = await axios.post('/api/upload-outfit', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60_000,
      })

      setDetectedItems(data.detectedItems || [])
      setUpgradeResult(data)
      setCurrentStep(3)

    } catch (err) {
      console.error('API error:', err)
      if (err.response?.status === 503 || err.code === 'ERR_NETWORK') {
        setError('The backend API is offline. Please start the Spring Boot server on port 8080.')
      } else {
        setError(err.response?.data?.message || err.message || 'An unexpected error occurred.')
      }
      setCurrentStep(0)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setImageFile(null)
    setPreviewUrl(null)
    setDetectedItems([])
    setUpgradeResult(null)
    setError(null)
    setCurrentStep(0)
  }

  // ── Demo mode: fetch recommendations without image ─────────────
  const handleDemoRecommend = async () => {
    setError(null)
    setLoading(true)
    try {
      const { data } = await axios.post('/api/recommend-outfit', { occasion, style: null })
      setUpgradeResult({
        occasion,
        detectedItems: [],
        overallCompatibilityScore: 0.75,
        upgradeSuggestions: [],
        itemsToReplace: [],
        itemsToAdd: [],
        upgradeSummary: `Browse our top picks for ${occasion} occasions below.`,
        recommendations: data.recommendations || [],
      })
      setCurrentStep(3)
    } catch (err) {
      setError('Could not load recommendations. Is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  // ── Render helpers ─────────────────────────────────────────────
  const isReady = imageFile && occasion

  return (
    <div className="app">

      {/* ── Header ─────────────────────────────────────────────── */}
      <header className="header">
        <div className="container">
          <div className="header-inner">
            <div className="logo">
              <div className="logo-icon">✨</div>
              <span className="logo-text">FashionAI</span>
              <span className="logo-badge">AI Powered</span>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
              {upgradeResult && (
                <button className="btn btn-ghost" onClick={handleReset} id="btn-reset">
                  ↩ New Analysis
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: '0 0 4rem' }}>
        <div className="container">

          {/* Hero */}
          <section className="hero">
            <div className="hero-eyebrow">
              <span>🤖</span> AI · Computer Vision · Style Intelligence
            </div>
            <h1 className="hero-title">
              Outfit Intelligence<br />Powered by AI
            </h1>
            <p className="hero-subtitle">
              Upload your outfit, pick an occasion, and let our AI detect clothing items,
              score compatibility, and suggest upgrades — all in seconds.
            </p>
          </section>

          {/* Stepper */}
          <div className="stepper" role="list" aria-label="Progress steps">
            {STEPS.map((step, i) => (
              <div key={step.id} style={{ display: 'flex', alignItems: 'center' }}>
                <div
                  role="listitem"
                  className={`step ${i === currentStep ? 'active' : i < currentStep ? 'done' : ''}`}
                >
                  <div className="step-bubble">
                    {i < currentStep ? '✓' : i + 1}
                  </div>
                  <span className="step-label">{step.label}</span>
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`step-connector ${i < currentStep ? 'done' : ''}`} />
                )}
              </div>
            ))}
          </div>

          {/* ── Loading overlay ───────────────────────────────── */}
          {loading && (
            <div className="card card-glow" style={{ marginBottom: '2rem' }}>
              <div className="loader-wrapper">
                <div className="loader-ring pulse" />
                <div className="loader-text">Processing your outfit…</div>
                <div className="loader-steps">
                  {LOADING_STEPS.map((s, i) => (
                    <div
                      key={i}
                      className={`loader-step ${i === loadingStep ? 'active' : i < loadingStep ? 'done' : ''}`}
                      style={{ animationDelay: `${i * 200}ms` }}
                    >
                      {i < loadingStep ? '✅' : i === loadingStep ? '⚡' : '○'} {s.text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Error banner ──────────────────────────────────── */}
          {error && !loading && (
            <div className="alert alert-error" id="error-banner" style={{ marginBottom: '1.5rem' }}>
              <span>⚠️</span>
              <div>
                <strong>Error</strong><br />{error}
              </div>
            </div>
          )}

          {/* ── Step 0 + 1: Upload & Occasion ────────────────── */}
          {!upgradeResult && !loading && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '1.5rem',
                marginBottom: '2rem',
              }}
            >
              {/* Upload card */}
              <div className="card">
                <div className="section-header">
                  <div>
                    <div className="section-title">📸 Upload Outfit</div>
                    <div className="section-subtitle">Clear, well-lit photos work best</div>
                  </div>
                </div>
                <ImageUploader
                  onImageSelect={handleImageSelect}
                  previewUrl={previewUrl}
                />
              </div>

              {/* Occasion card */}
              <div className="card">
                <div className="section-header">
                  <div>
                    <div className="section-title">🎯 Target Occasion</div>
                    <div className="section-subtitle">Where are you wearing this?</div>
                  </div>
                </div>
                <OccasionSelector selected={occasion} onSelect={setOccasion} />

                <div className="divider" />

                {/* CTA */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  <button
                    id="btn-analyse"
                    className="btn btn-primary"
                    onClick={handleAnalyse}
                    disabled={!isReady}
                    style={{ width: '100%', justifyContent: 'center', fontSize: '1rem', padding: '0.9rem' }}
                  >
                    <span>🔍</span>
                    Analyse My Outfit
                  </button>
                  <button
                    id="btn-demo"
                    className="btn btn-ghost"
                    onClick={handleDemoRecommend}
                    style={{ width: '100%', justifyContent: 'center' }}
                  >
                    <span>💡</span>
                    Browse Recommendations (Demo)
                  </button>
                </div>

                {/* Feature list */}
                <div style={{ marginTop: '1.5rem' }}>
                  {[
                    ['🎯', 'YOLO clothing detection'],
                    ['🧠', 'ResNet50 feature extraction'],
                    ['📊', 'Cosine similarity ranking'],
                    ['✨', 'Rule-based outfit upgrade'],
                  ].map(([icon, text]) => (
                    <div
                      key={text}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.6rem',
                               fontSize: '0.82rem', color: 'var(--text-muted)',
                               marginBottom: '0.4rem' }}
                    >
                      {icon} {text}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── Results: Detection + Upgrade + Recommendations ── */}
          {upgradeResult && !loading && (
            <div className="animate-fade-up">

              {/* Detection results */}
              {upgradeResult.detectedItems?.length > 0 && (
                <div className="card" style={{ marginBottom: '1.5rem' }}>
                  <div className="section-header">
                    <div>
                      <div className="section-title">🔍 Detected Clothing</div>
                      <div className="section-subtitle">Items found in your outfit</div>
                    </div>
                    <span className="tag tag-green">
                      {upgradeResult.detectedItems.length} items
                    </span>
                  </div>
                  <DetectionResults items={upgradeResult.detectedItems} />
                </div>
              )}

              {/* Upgrade analysis */}
              <div>
                <div className="section-header">
                  <div>
                    <div className="section-title">✨ Outfit Upgrade</div>
                    <div className="section-subtitle">
                      Occasion: <strong style={{ color: 'var(--accent-1)', textTransform: 'capitalize' }}>
                        {upgradeResult.occasion}
                      </strong>
                    </div>
                  </div>
                  <button className="btn btn-secondary" onClick={handleReset} id="btn-try-another">
                    ↩ Try Another
                  </button>
                </div>
                <UpgradePanel result={upgradeResult} />
              </div>
            </div>
          )}

        </div>
      </main>

      {/* ── Footer ─────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="container">
          <p>
            AI Fashion Recommender · Built with Spring Boot + React · Powered by YOLOv8 & ResNet50
          </p>
        </div>
      </footer>
    </div>
  )
}
