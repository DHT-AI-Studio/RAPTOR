import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import { useTheme } from '../contexts/ThemeContext.jsx'
import { apiLogin } from '../api/client.js'

export default function LoginPage({ sessionExpired = false }) {
  const { login } = useAuth()
  const { themeKey } = useTheme()
  const [form, setForm] = useState({ username: '', password: '', realmName: 'dhtsolution' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [apiBase, setApiBase] = useState(() => localStorage.getItem('raptor_api_base') || '')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    localStorage.setItem('raptor_api_base', apiBase.trim())
    try {
      const data = await apiLogin(form)
      const token = data.access_token
      let userInfo = { sub: 'unknown' }
      try {
        const payload = JSON.parse(atob(token.split('.')[1]))
        userInfo = { sub: payload.sub || payload.preferred_username || 'user', ...payload }
      } catch {}
      login(token, userInfo)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4"
      style={{ background: 'var(--bg-base)' }}>

      <div className="w-full max-w-sm">

        {/* Logo area */}
        <div className="text-center mb-10">
          <div className="inline-block mb-4">
            <img src="/raptor-logo.png" alt="Raptor" className="w-20 h-20 object-contain mx-auto"
              style={themeKey === 'light'
                ? { mixBlendMode: 'multiply' }
                : { filter: 'brightness(0) invert(1)', opacity: 0.9 }
              } />
          </div>
          <h1 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--text-primary)' }}>
            Raptor
          </h1>
          <p className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>
            AI-Powered Knowledge Platform
          </p>
        </div>

        {/* Session expired */}
        {sessionExpired && (
          <div className="rounded-xl px-4 py-3 text-sm mb-5 text-center"
            style={themeKey === 'light'
              ? { background: 'rgba(180,83,9,0.07)', border: '1px solid rgba(180,83,9,0.25)', color: '#92400e' }
              : { background: 'rgba(234,179,8,0.08)', border: '1px solid rgba(234,179,8,0.22)', color: '#fde047' }}>
            登入逾期，請重新登入。
          </div>
        )}

        {/* Card */}
        <div className="card p-7">
          {/* Card top accent line */}
          <div className="absolute top-0 left-1/2 -translate-x-1/2 w-24 h-px"
            style={{ background: 'linear-gradient(to right, transparent, var(--accent), transparent)' }} />

          <h2 className="text-base font-semibold mb-6 tracking-tight" style={{ color: 'var(--text-primary)' }}>
            Sign In
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                Username
              </label>
              <input className="input-field" type="text" placeholder="Enter username"
                value={form.username} onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                required autoFocus />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                Password
              </label>
              <input className="input-field" type="password" placeholder="Enter password"
                value={form.password} onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                required />
            </div>

            <button type="button"
              className="flex items-center gap-1.5 text-xs transition-colors"
              style={{ color: 'var(--text-muted)' }}
              onMouseEnter={e => e.currentTarget.style.color = 'var(--text-secondary)'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
              onClick={() => setShowAdvanced(v => !v)}>
              <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5"
                className={`w-3 h-3 transition-transform duration-150 ${showAdvanced ? 'rotate-90' : ''}`}>
                <path strokeLinecap="round" d="M4 2l4 4-4 4" />
              </svg>
              Advanced Settings
            </button>

            {showAdvanced && (
              <div className="space-y-3 pt-1">
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>Realm Name</label>
                  <input className="input-field" value={form.realmName}
                    onChange={e => setForm(f => ({ ...f, realmName: e.target.value }))} />
                </div>
                <div>
                  <label className="block text-xs font-medium mb-1.5" style={{ color: 'var(--text-secondary)' }}>
                    API Base URL
                    <span className="ml-1.5 font-normal text-[11px]" style={{ color: 'var(--text-muted)' }}>(empty = same origin)</span>
                  </label>
                  <input className="input-field" placeholder="http://192.168.x.x:8012"
                    value={apiBase} onChange={e => setApiBase(e.target.value)} />
                </div>
              </div>
            )}

            {error && <div className="error-box">{error}</div>}

            <button type="submit" disabled={loading} className="btn-primary w-full py-2.5 mt-2">
              {loading ? (
                <span className="flex items-center gap-2">
                  <Spinner /> Signing in…
                </span>
              ) : 'Sign In'}
            </button>
          </form>
        </div>

        <div className="text-center mt-6 space-y-1">
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>© 2026 DHT Solution. All rights reserved.</p>
          <p className="text-[11px]" style={{ color: 'var(--text-muted)' }}>
            <a href="#" className="hover:underline transition-colors" style={{ color: 'var(--text-muted)' }}>Privacy Policy</a>
            <span className="mx-1.5">|</span>
            <a href="#" className="hover:underline transition-colors" style={{ color: 'var(--text-muted)' }}>Terms of Service</a>
          </p>
        </div>
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
    </svg>
  )
}
