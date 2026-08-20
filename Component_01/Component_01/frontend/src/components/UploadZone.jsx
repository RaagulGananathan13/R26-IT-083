import { useRef, useState } from 'react'

export default function UploadZone({ onUpload, isLoading }) {
  const [dragover, setDragover] = useState(false)
  const [preview, setPreview] = useState(null)
  const fileRef = useRef()

  const handleFile = (file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG)')
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    onUpload(file)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragover(false)
    handleFile(e.dataTransfer.files[0])
  }

  const onDragOver = (e) => { e.preventDefault(); setDragover(true) }
  const onDragLeave = () => setDragover(false)
  const onClick = () => fileRef.current?.click()
  const onFileChange = (e) => handleFile(e.target.files[0])

  return (
    <div className="card upload-card">
      <div>
        <div className="card-title">Upload chest X ray</div>
        <div className="card-subtitle">PNG or JPG. Frontal view recommended for best results.</div>
      </div>

      {preview ? (
        <div style={{ position: 'relative' }}>
          <div className="preview-frame">
            <img src={preview} alt="Uploaded X ray" />
          </div>
          {!isLoading && (
            <button
              onClick={() => { setPreview(null); fileRef.current.value = '' }}
              className="link-button"
            >
              Choose another image
            </button>
          )}
          {isLoading && (
            <div className="loading-overlay">
              <div style={{ textAlign: 'center' }}>
                <div className="spinner" />
                <div>Analyzing image</div>
                <div style={{ fontSize: '0.8rem', opacity: 0.7, marginTop: 4 }}>Running both models</div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div
          className={`dropzone ${dragover ? 'is-dragover' : ''}`}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onClick={onClick}
          id="upload-dropzone"
        >
          <div className="dropzone-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M12 16V4" />
              <path d="m6 10 6-6 6 6" />
              <path d="M4 20h16" />
            </svg>
          </div>
          <div className="dropzone-title">Drop the image here</div>
          <div className="dropzone-subtitle">or click to browse files</div>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        onChange={onFileChange}
        className="hidden"
        id="file-input"
      />
    </div>
  )
}
