/**
 * RecommendationCard — displays a single outfit recommendation with
 * score badge, item tags, color palette, and reason text.
 */
export default function RecommendationCard({ recommendation, index }) {
  const { outfit, similarityScore, rankingScore, rank, recommendationReason } = recommendation

  const scorePercent = Math.round((rankingScore || similarityScore || 0) * 100)

  const getScoreColor = (score) => {
    if (score >= 80) return 'var(--success)'
    if (score >= 60) return 'var(--accent-1)'
    if (score >= 40) return 'var(--warning)'
    return 'var(--error)'
  }

  return (
    <div
      id={`recommendation-card-${index}`}
      className="reco-card animate-fade-up"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      {/* Header: rank + score badge */}
      <div className="reco-card-header">
        <div className="reco-rank">#{rank || index + 1}</div>
        <div
          className="reco-score-badge"
          style={{ color: getScoreColor(scorePercent) }}
        >
          {scorePercent}% match
        </div>
      </div>

      {/* Outfit name + style */}
      <div className="reco-name">{outfit?.name || 'Outfit'}</div>
      <div className="reco-style">
        <span className="tag tag-blue" style={{ marginRight: '0.35rem' }}>
          {outfit?.style || 'mixed'}
        </span>
        {outfit?.season && outfit.season !== 'all' && (
          <span className="tag tag-purple">{outfit.season}</span>
        )}
      </div>

      {/* Occasions */}
      {outfit?.occasions?.length > 0 && (
        <div className="reco-items" style={{ marginBottom: '0.5rem' }}>
          {outfit.occasions.slice(0, 3).map((occ, i) => (
            <span key={i} className="tag tag-pink">{occ}</span>
          ))}
        </div>
      )}

      {/* Clothing items */}
      {outfit?.items?.length > 0 && (
        <div className="reco-items">
          {outfit.items.slice(0, 5).map((item, i) => (
            <span key={i} className="reco-item-tag">
              {item.category || item}
            </span>
          ))}
        </div>
      )}

      {/* Color palette */}
      {outfit?.colorPalette?.length > 0 && (
        <div className="reco-colors">
          {outfit.colorPalette.slice(0, 5).map((color, i) => (
            <div
              key={i}
              title={color}
              style={{
                width: 14, height: 14,
                borderRadius: '50%',
                backgroundColor: `var(--color-${color?.replace(/\s+/g, '-')}, #6b7280)`,
                border: '1px solid rgba(255,255,255,0.15)',
                background: color.includes('blue') ? '#3b82f6' :
                            color.includes('white') ? '#f8fafc' :
                            color.includes('black') ? '#1e1e2e' :
                            color.includes('grey') ? '#6b7280' :
                            color.includes('red') ? '#ef4444' :
                            color.includes('green') ? '#10b981' :
                            color.includes('navy') ? '#1e3a5f' :
                            color.includes('brown') ? '#92400e' :
                            color.includes('pink') || color.includes('blush') ? '#ec4899' :
                            color.includes('purple') ? '#a855f7' :
                            color.includes('gold') ? '#f59e0b' : '#6b7280'
              }}
            />
          ))}
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: '0.25rem' }}>
            {outfit.colorPalette.slice(0, 3).join(', ')}
          </span>
        </div>
      )}

      {/* Recommendation reason */}
      {recommendationReason && (
        <div className="reco-reason">"{recommendationReason}"</div>
      )}
    </div>
  )
}
