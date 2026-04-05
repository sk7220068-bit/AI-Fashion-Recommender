/** Maps category names to emojis for visual decoration */
const CATEGORY_ICONS = {
  't-shirt':    '👕', 'shirt':     '👔', 'jacket':   '🧥',
  'jeans':      '👖', 'trousers':  '👖', 'shorts':   '🩳',
  'dress':      '👗', 'skirt':     '👗', 'coat':     '🧥',
  'sneakers':   '👟', 'boots':     '🥾', 'heels':    '👠',
  'blazer':     '🧥', 'hoodie':    '🧥', 'sweater':  '🧶',
  'handbag':    '👜', 'backpack':  '🎒', 'tie':      '👔',
  'camisole':   '👙', 'vest':      '🦺', 'default':  '👗',
}

const getCategoryIcon = (category) =>
  CATEGORY_ICONS[category?.toLowerCase()] || CATEGORY_ICONS.default

const COLOR_MAP = {
  'white':      '#f8fafc', 'black':      '#1e1e2e',
  'navy blue':  '#1e3a5f', 'blue':       '#3b82f6',
  'red':        '#ef4444', 'green':      '#10b981',
  'grey':       '#6b7280', 'yellow':     '#f59e0b',
  'orange':     '#f97316', 'purple':     '#a855f7',
  'pink':       '#ec4899', 'brown':      '#92400e',
  'navy':       '#1e3a5f', 'beige':      '#d4b896',
}

const getColorHex = (colorName) =>
  COLOR_MAP[colorName?.toLowerCase()] || '#6b7280'

/**
 * DetectionResults — displays detected clothing items as styled chips
 * with confidence bars, dominant color swatches, and style tags.
 */
export default function DetectionResults({ items }) {
  if (!items || items.length === 0) {
    return (
      <div className="empty-state">
        <span className="empty-state-icon">🔍</span>
        <h3>No clothing detected</h3>
        <p>Try uploading a clear, well-lit photo of your outfit.</p>
      </div>
    )
  }

  return (
    <div>
      <div
        style={{ marginBottom: '0.75rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}
      >
        {items.length} item{items.length !== 1 ? 's' : ''} detected
      </div>
      <div className="detection-grid">
        {items.map((item, idx) => (
          <div
            key={idx}
            id={`detected-item-${idx}`}
            className="detection-chip animate-fade-up"
            style={{ animationDelay: `${idx * 80}ms` }}
          >
            <span className="detection-icon">{getCategoryIcon(item.category)}</span>

            <div className="detection-category">{item.category}</div>

            {/* Style tag */}
            {item.style && (
              <span className="tag tag-purple" style={{ fontSize: '0.65rem' }}>
                {item.style}
              </span>
            )}

            {/* Confidence bar */}
            <div className="detection-confidence">
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${Math.round((item.confidence || 0) * 100)}%` }}
                />
              </div>
              {Math.round((item.confidence || 0) * 100)}%
            </div>

            {/* Dominant color swatch */}
            {item.dominantColor && item.dominantColor !== 'unknown' && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <span
                  className="color-dot"
                  style={{ backgroundColor: getColorHex(item.dominantColor) }}
                  title={item.dominantColor}
                />
                <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', textTransform: 'capitalize' }}>
                  {item.dominantColor}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
