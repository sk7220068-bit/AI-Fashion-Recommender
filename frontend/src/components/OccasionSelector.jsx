const OCCASIONS = [
  { id: 'casual',  label: 'Casual',  icon: '👟', description: 'Everyday relaxed wear' },
  { id: 'work',    label: 'Work',    icon: '💼', description: 'Office & professional' },
  { id: 'party',   label: 'Party',   icon: '🎉', description: 'Evening & celebrations' },
  { id: 'date',    label: 'Date',    icon: '💕', description: 'Romantic evenings' },
  { id: 'formal',  label: 'Formal',  icon: '🎩', description: 'Black tie & galas' },
  { id: 'sport',   label: 'Sport',   icon: '🏃', description: 'Athletic activities' },
]

/**
 * OccasionSelector — pill-button selector for the target occasion.
 * Highlights the active selection with a gradient style.
 */
export default function OccasionSelector({ selected, onSelect }) {
  return (
    <div>
      <div className="occasion-grid" role="group" aria-label="Select occasion">
        {OCCASIONS.map((occ) => (
          <button
            key={occ.id}
            id={`occasion-${occ.id}`}
            className={`occasion-pill ${selected === occ.id ? 'selected' : ''}`}
            onClick={() => onSelect(occ.id)}
            title={occ.description}
            aria-pressed={selected === occ.id}
          >
            <span>{occ.icon}</span>
            <span>{occ.label}</span>
          </button>
        ))}
      </div>
      {selected && (
        <p style={{ marginTop: '0.75rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
          ✓ Optimizing suggestions for <strong style={{ color: 'var(--accent-1)' }}>{selected}</strong> occasions
        </p>
      )}
    </div>
  )
}
