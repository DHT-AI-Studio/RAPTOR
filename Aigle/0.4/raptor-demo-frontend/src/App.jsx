import React, { useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext.jsx'
import { ThemeProvider } from './contexts/ThemeContext.jsx'
import LoginPage from './components/LoginPage.jsx'
import Layout from './components/Layout.jsx'
import SearchPanel from './components/SearchPanel.jsx'
import ChatPanel from './components/ChatPanel.jsx'
import UploadPanel from './components/UploadPanel.jsx'
import VideoSearchPanel from './components/VideoSearchPanel.jsx'
import HistoryPanel from './components/HistoryPanel.jsx'

function AppContent() {
  const { isAuthenticated, sessionExpired } = useAuth()
  const [activeTab, setActiveTab] = useState('upload')

  if (!isAuthenticated) return <LoginPage sessionExpired={sessionExpired} />

  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab}>
      <div className={`flex-1 min-h-0 flex flex-col ${activeTab === 'search'  ? '' : 'hidden'}`}><SearchPanel /></div>
      <div className={`flex-1 min-h-0 flex flex-col ${activeTab === 'chat'    ? '' : 'hidden'}`}><ChatPanel /></div>
      <div className={`flex-1 min-h-0 flex flex-col ${activeTab === 'upload'  ? '' : 'hidden'}`}><UploadPanel /></div>
      <div className={`flex-1 min-h-0 flex flex-col ${activeTab === 'video'   ? '' : 'hidden'}`}><VideoSearchPanel /></div>
      <div className={`flex-1 min-h-0 flex flex-col ${activeTab === 'history' ? '' : 'hidden'}`}><HistoryPanel /></div>
    </Layout>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      {/* Background ambient orbs */}
      <div className="bg-orb w-[600px] h-[600px] -top-64 -left-48"
        style={{ background: `radial-gradient(circle, var(--orb1), transparent 65%)` }} />
      <div className="bg-orb w-[500px] h-[500px] bottom-0 right-0"
        style={{ background: `radial-gradient(circle, var(--orb2), transparent 65%)` }} />

      <div className="relative z-10 h-screen">
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </div>
    </ThemeProvider>
  )
}
