import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DiagnosisPage from './pages/DiagnosisPage'
import ResultPage from './pages/ResultPage'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <div className="header-inner">
            <div className="header-logo">🏥</div>
            <div>
              <div className="header-title">智能医疗诊断辅助系统</div>
              <div className="header-subtitle">RAG + Deep Learning · AI-Powered Diagnosis</div>
            </div>
          </div>
        </header>

        <main className="app-main">
          <Routes>
            <Route path="/" element={<DiagnosisPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </main>

        <footer className="app-footer">
          © 2026 智能医疗诊断辅助系统 | 仅供学习参考，不构成医疗建议
        </footer>
      </div>
    </BrowserRouter>
  )
}

export default App
