import RecommendationCard from './RecommendationCard'

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
export default function UpgradePanel({ result }) {
  if (!result) return null

  const {
    overallCompatibilityScore,
    itemsToReplace,
    itemsToAdd,
    upgradeSuggestions,
    upgradeSummary,
    aiGeneratedDescription,
    recommendations,
  } = result

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
