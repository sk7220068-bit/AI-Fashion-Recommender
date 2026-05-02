import { useEffect, useState } from 'react'
import axios from 'axios'

const splitCsv = (v) => v.split(',').map(s => s.trim()).filter(Boolean)

export default function UserProfilePanel({ userId, onUserIdChange }) {
  const [open, setOpen] = useState(false)
  const [styles, setStyles] = useState('casual')
  const [colors, setColors] = useState('white, navy')
  const [avoid, setAvoid] = useState('')
  const [status, setStatus] = useState('')

  useEffect(() => {
    if (!userId) return
    axios.get(`/api/user-preferences/${userId}`)
      .then(({ data }) => {
        setStyles((data.preferredStyles || []).join(', '))
        setColors((data.preferredColors || []).join(', '))
        setAvoid((data.avoidedItems || []).join(', '))
      })
      .catch(() => {})
  }, [userId])

  const save = async () => {
    if (!userId) return
    setStatus('Saving...')
    try {
      await axios.post('/api/user-preferences', {
        userId,
        preferredStyles: splitCsv(styles),
        preferredColors: splitCsv(colors),
        avoidedItems: splitCsv(avoid),
      })
      setStatus('Saved')
    } catch {
      setStatus('Could not save profile')
    }
  }

  return (
    <>
      <button className="btn btn-ghost" onClick={() => setOpen(!open)}>
        👤 Profile
      </button>
      {open && (
        <div className="card" style={{ position: 'absolute', top: '100%', right: '1rem', marginTop: '0.75rem', display: 'grid', gap: '0.6rem', width: 320, zIndex: 10, backgroundColor: 'var(--surface-50)' }}>
          <label>
            User ID
            <input value={userId} onChange={(e) => onUserIdChange(e.target.value)} className="input" />
          </label>
          <label>
            Preferred Styles (csv)
            <input value={styles} onChange={(e) => setStyles(e.target.value)} className="input" />
          </label>
          <label>
            Preferred Colors (csv)
            <input value={colors} onChange={(e) => setColors(e.target.value)} className="input" />
          </label>
          <label>
            Avoid Items (csv)
            <input value={avoid} onChange={(e) => setAvoid(e.target.value)} className="input" />
          </label>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--accent-1)' }}>{status}</span>
            <button className="btn btn-primary" onClick={save}>Save</button>
          </div>
        </div>
      )}
    </>
  )
}
