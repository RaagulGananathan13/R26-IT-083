export default function Header() {
  return (
    <header className="site-header">
      <div className="brand">
        <div className="brand-mark">CV</div>
        <div>
          <div className="brand-title">CardioVision</div>
          <div className="brand-subtitle">Cardiomegaly detection and report generation</div>
        </div>
      </div>
      <div className="status-pill">
        <span className="status-dot" />
        Model online
      </div>
    </header>
  )
}
