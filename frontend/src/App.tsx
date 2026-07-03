import { BrowserRouter, Routes, Route } from 'react-router-dom'
import DiagnosisPage from './pages/DiagnosisPage'
import ResultPage from './pages/ResultPage'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <h1>🏥 智能医疗诊断辅助系统</h1>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<DiagnosisPage />} />
            <Route path="/result" element={<ResultPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
