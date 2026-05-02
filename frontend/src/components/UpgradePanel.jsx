import RecommendationCard from './RecommendationCard'
import axios from 'axios'
import { useRenderJobPolling } from '../hooks/useRenderJobPolling'

/** Circular progress ring for compatibility score */
function CompatRing({ score }) {
  const radius = 50
  const circumference = 2 * Math.PI * radius
  const pct = Math.max(0, Math.min(1, score))
  const offset = circumference * (1 - pct)
  const displayPct = Math.round(pct * 100)

  // Color: green ≥ 70%, amber ≥ 50%, red < 50%
  const strokeColor = pct >= 0.70 ? '#10b981' : pct >= 0.50 ? '#f59e0b' : '#ef4444'

  const label =
    pct >= 0.85 ? 'Excellent' :
    pct >= 0.70 ? 'Good' :
    pct >= 0.50 ? 'Fair' : 'Poor'

  return (
    <div className="compat-ring-wrapper">
      <div className="compat-ring">
        <svg viewBox="0 0 120 120" width="120" height="120">
          <circle className="track" cx="60" cy="60" r={radius} />
          <circle
            className="fill"
            cx="60" cy="60" r={radius}
            stroke={strokeColor}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="compat-score-label">
          <span className="compat-score-num" style={{ color: strokeColor }}>
            {displayPct}%
          </span>
          <span className="compat-score-sub">compat.</span>
        </div>
      </div>
      <span className="compat-label" style={{ color: strokeColor }}>{label}</span>
    </div>
  )
}

/**
 * UpgradePanel — renders the full outfit upgrade analysis:
 *   - Compatibility ring
 *   - Items to replace
 *   - Items to add
 *   - Specific upgrade suggestions
 *   - Summary paragraph
 *   - Outfit recommendations
 */
export default function UpgradePanel({ result, userId }) {
  if (!result) return null

  const {
    overallCompatibilityScore,
    itemsToReplace,
    itemsToAdd,
    upgradeSuggestions,
    upgradeSummary,
    aiGeneratedDescription,
    recommendations,
    upgradedImageUrl,
    upgradedImageAlternatives,
    renderStatus,
  } = result

  const submitFeedback = async (recommendation, action) => {
    try {
      await axios.post('/api/recommend-feedback', {
        userId,
        action,
        outfitId: recommendation?.outfit?.id,
        style: recommendation?.outfit?.style,
        occasion: result?.occasion,
      })
    } catch (e) {
      console.warn('feedback failed', e)
    }
  }

  const { status: polledStatus, result: polledJobResult } = useRenderJobPolling(
    result?.renderJobId,
    result?.renderStatus || 'queued',
    (data) => console.log('Render job completed', data)
  )

  const currentRenderStatus = polledStatus || renderStatus
  const currentUpgradedImageUrl = polledJobResult?.mainImageUrl || upgradedImageUrl
  const currentUpgradedImageAlternatives = polledJobResult?.variants || upgradedImageAlternatives

  return (
    <div className="animate-fade-up">

      {/* ── Compatibility Ring + Summary ────────────────────────── */}
      <div
        className="card card-glow"
        style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start',
                 flexWrap: 'wrap', marginBottom: '1.5rem' }}
      >
        <CompatRing score={overallCompatibilityScore || 0} />

        <div style={{ flex: 1, minWidth: 220 }}>
          <div
            className="section-title"
            style={{ marginBottom: '0.75rem' }}
          >
            Outfit Analysis
          </div>
          {upgradeSummary && (
            <div className="upgrade-summary">{upgradeSummary}</div>
          )}
          {aiGeneratedDescription && aiGeneratedDescription !== upgradeSummary && (
            <div
              className="upgrade-summary"
              style={{ marginTop: '0.75rem', background: 'rgba(59,130,246,0.08)',
                       borderColor: 'rgba(59,130,246,0.30)' }}
            >
              🤖 {aiGeneratedDescription}
            </div>
          )}
        </div>
      </div>

      {/* ── Upgraded Preview ───────────────────────────────────── */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div className="upgrade-section-title">
          <span>🖼️</span> Upgraded Preview
        </div>
        {currentUpgradedImageUrl ? (
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            <img
              src={currentUpgradedImageUrl}
              alt="Upgraded outfit preview"
              style={{ width: '100%', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)' }}
            />
            {!!currentUpgradedImageAlternatives?.length && (
              <div style={{ display: 'grid', gap: '0.6rem', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}>
                {currentUpgradedImageAlternatives.map((img, i) => (
                  <img key={i} src={img} alt={`Upgrade variant ${i + 1}`}
                    style={{ width: '100%', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.1)' }} />
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="upgrade-summary" style={{ marginTop: '0.5rem' }}>
            {(currentRenderStatus === 'pending' || currentRenderStatus === 'queued' || currentRenderStatus === 'running') && 'Rendering preview...'}
            {currentRenderStatus === 'failed' && 'Preview unavailable. Showing text-based upgrade only.'}
            {!currentRenderStatus && 'Preview will appear after rendering.'}
          </div>
        )}
      </div>

      {/* ── Upgrade Suggestions ─────────────────────────────────── */}
      {upgradeSuggestions?.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="upgrade-section-title">
            <span>✨</span> Specific Upgrade Suggestions
          </div>
          <div className="upgrade-list">
            {upgradeSuggestions.map((s, i) => (
              <div key={i} className="upgrade-item replace-item">
                <span className="upgrade-item-icon">💡</span>
                <span>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Items to Replace + Add ──────────────────────────────── */}
      {(itemsToReplace?.length > 0 || itemsToAdd?.length > 0) && (
        <div
          style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                   gap: '1rem', marginBottom: '1.5rem' }}
        >
          {itemsToReplace?.length > 0 && (
            <div className="card">
              <div className="upgrade-section-title">
                <span>🔄</span> Items to Replace
              </div>
              <div className="upgrade-list">
                {itemsToReplace.map((item, i) => (
                  <div key={i} className="upgrade-item replace-item">
                    <span className="upgrade-item-icon">⚠️</span>
                    <span style={{ textTransform: 'capitalize' }}>{item}</span>
                    <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--warning)' }}>
                      Not ideal
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {itemsToAdd?.length > 0 && (
            <div className="card">
              <div className="upgrade-section-title">
                <span>➕</span> Items to Add
              </div>
              <div className="upgrade-list">
                {itemsToAdd.map((item, i) => (
                  <div key={i} className="upgrade-item add-item">
                    <span className="upgrade-item-icon">✅</span>
                    <span style={{ textTransform: 'capitalize' }}>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Outfit Recommendations ──────────────────────────────── */}
      {recommendations?.length > 0 && (
        <div>
          <div className="section-header">
            <div>
              <div className="section-title">You Might Also Like</div>
              <div className="section-subtitle">
                Outfits that match your style and occasion
              </div>
            </div>
            <span className="tag tag-purple">{recommendations.length} outfits</span>
          </div>
          <div className="reco-grid">
            {recommendations.map((rec, i) => (
              <RecommendationCard
                key={rec.outfit?.id || i}
                recommendation={rec}
                index={i}
                onFeedback={submitFeedback}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── No suggestions fallback ─────────────────────────────── */}
      {!upgradeSuggestions?.length && !itemsToReplace?.length && !itemsToAdd?.length && (
        <div className="card">
          <div className="empty-state">
            <span className="empty-state-icon">🎉</span>
            <h3>Your outfit looks great!</h3>
            <p>No improvements needed for this occasion.</p>
          </div>
        </div>
      )}
    </div>
  )
}
