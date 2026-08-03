import { useState } from 'react'

export default function GradCamViewer({ originalImg, heatmapImg }) {
  const [viewMode, setViewMode] = useState('overlay') // original, heatmap, overlay

  return (
    <div className="card panel">
      <div className="panel-header">
        <div>
          <div className="eyebrow">Explainable AI</div>
          <div className="panel-title">GradCAM focus map</div>
        </div>
        <div className="segmented">
          <button
            onClick={() => setViewMode('original')}
            className={`segmented-btn ${viewMode === 'original' ? 'active' : ''}`}
          >
            Original
          </button>
          <button
            onClick={() => setViewMode('overlay')}
            className={`segmented-btn ${viewMode === 'overlay' ? 'active' : ''}`}
          >
            Overlay
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`segmented-btn ${viewMode === 'heatmap' ? 'active' : ''}`}
          >
            Heatmap
          </button>
        </div>
      </div>

      <div className="image-frame">
        <img
          src={originalImg}
          alt="Original X ray"
          style={{ opacity: viewMode === 'heatmap' ? 0 : 1, transition: 'opacity 0.3s' }}
        />
        <img
          src={heatmapImg}
          alt="GradCAM heatmap"
          className="absolute inset-0"
          style={{
            opacity: viewMode === 'original' ? 0 : (viewMode === 'heatmap' ? 1 : 0.7),
            transition: 'opacity 0.3s',
            width: '100%',
            height: '100%',
            objectFit: 'contain',
            mixBlendMode: 'screen'
          }}
        />
      </div>

      <div className="panel-note">
        GradCAM highlights regions that contributed most to the prediction.
      </div>
    </div>
  )
}
