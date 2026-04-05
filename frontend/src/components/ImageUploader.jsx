import { useRef, useState } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']

/**
 * ImageUploader — drag-and-drop or click-to-browse image upload component.
 * Emits the selected File and a local preview URL to the parent.
 */
export default function ImageUploader({ onImageSelect, previewUrl }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)

  const handleFile = (file) => {
    setError(null)
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setError('Please upload a JPG, PNG, or WebP image.')
      return
    }
    if (file.size > 20 * 1024 * 1024) {
      setError('Image must be under 20 MB.')
      return
    }
    const url = URL.createObjectURL(file)
    onImageSelect(file, url)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleChange = (e) => handleFile(e.target.files[0])

  if (previewUrl) {
    return (
      <div className="upload-preview animate-fade-up">
        <img src={previewUrl} alt="Uploaded outfit" />
        <div className="upload-preview-overlay">
          <button
            className="upload-change-btn"
            onClick={() => {
              onImageSelect(null, null)
              if (inputRef.current) inputRef.current.value = ''
            }}
            aria-label="Change image"
          >
            🔄 Change Image
          </button>
        </div>
      </div>
    )
  }

  return (
    <div>
      <div
        id="image-upload-zone"
        role="button"
        tabIndex={0}
        aria-label="Upload an outfit image"
        className={`upload-zone ${dragging ? 'dragover' : ''}`}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <span className="upload-icon">👗</span>
        <div className="upload-title">Drop your outfit photo here</div>
        <div className="upload-hint">
          Click to browse or drag & drop · JPG, PNG, WebP · Max 20 MB
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>
          <span>⚠️</span> {error}
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        id="image-file-input"
        accept="image/jpeg,image/png,image/webp"
        style={{ display: 'none' }}
        onChange={handleChange}
      />
    </div>
  )
}
