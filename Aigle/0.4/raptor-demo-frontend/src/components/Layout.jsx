import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import { useTheme, THEMES } from '../contexts/ThemeContext.jsx'
import { apiLogout } from '../api/client.js'

const NAV_ITEMS = [
  // Demo mode: Search & Chat hidden
  // { id: 'search', label: 'Search', icon: <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-[17px] h-[17px]"><circle cx="8.5" cy="8.5" r="5.2"/><path strokeLinecap="round" d="M12.5 12.5L16.5 16.5"/></svg> },
  // { id: 'chat',   label: 'Chat',   icon: <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-[17px] h-[17px]"><path strokeLinecap="round" strokeLinejoin="round" d="M2.5 5.5a2 2 0 012-2h11a2 2 0 012 2v6a2 2 0 01-2 2H6.5l-4 3V5.5z"/></svg> },
  {
    id: 'upload', label: 'File Upload',
    icon: <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-[17px] h-[17px]"><path strokeLinecap="round" strokeLinejoin="round" d="M10 13V4m0 0L7 7m3-3l3 3"/><path strokeLinecap="round" d="M3 14.5v.5a3 3 0 003 3h8a3 3 0 003-3v-.5"/></svg>,
  },
  {
    id: 'video', label: 'Video Search',
    icon: <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-[17px] h-[17px]"><rect x="2" y="4.5" width="12" height="11" rx="1.5"/><path strokeLinecap="round" strokeLinejoin="round" d="M14 8l4-2.5v7L14 10.5"/></svg>,
  },
  {
    id: 'history', label: 'Upload History',
    icon: <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-[17px] h-[17px]"><circle cx="10" cy="10" r="7.5"/><path strokeLinecap="round" strokeLinejoin="round" d="M10 6v4.5l2.5 1.5"/></svg>,
  },
]

export default function Layout({ activeTab, setActiveTab, children }) {
  const { user, token, logout } = useAuth()
  const { themeKey, setThemeKey } = useTheme()
  const [collapsed, setCollapsed] = useState(false)

  const handleLogout = async () => {
    try { await apiLogout(token) } catch {}
    logout()
  }

  const initials = (user?.preferred_username || user?.sub || '?')[0].toUpperCase()
  const displayName = user?.preferred_username || user?.sub || 'User'

  return (
    <div className="flex h-screen overflow-hidden">

      {/* ── Sidebar ── */}
      <aside
        className="flex-shrink-0 flex flex-col relative transition-all duration-200"
        style={{
          width: collapsed ? 56 : 224,
          background: 'var(--glass-bg)',
          borderRight: '1px solid var(--border)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
        }}
      >
        {/* Logo */}
        <div className="px-3 py-4 flex items-center gap-2 overflow-hidden" style={{ borderBottom: '1px solid var(--border)', minHeight: 60 }}>
          {/* Hamburger toggle */}
          <button
            onClick={() => setCollapsed(v => !v)}
            className="w-8 h-8 flex-shrink-0 flex flex-col items-center justify-center gap-[5px] rounded-lg transition-colors"
            style={{ color: 'var(--text-muted)' }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            title={collapsed ? 'Expand' : 'Collapse'}
          >
            <span className="block w-4 h-[1.5px] rounded-full bg-current transition-all duration-200"
              style={{ transformOrigin: 'center', transform: collapsed ? 'rotate(0)' : 'rotate(0)' }}/>
            <span className="block h-[1.5px] rounded-full bg-current transition-all duration-200"
              style={{ width: collapsed ? 12 : 16 }}/>
            <span className="block w-4 h-[1.5px] rounded-full bg-current"/>
          </button>

          {!collapsed && (
            <>
              <div className="w-10 h-10 flex-shrink-0 flex items-center justify-center">
                <img src="/raptor-logo.png" alt="Raptor" className="w-10 h-10 object-contain"
                  style={themeKey === 'light'
                    ? { mixBlendMode: 'multiply' }
                    : { filter: 'invert(1) brightness(1.15)', mixBlendMode: 'screen' }
                  } />
              </div>
              <div className="overflow-hidden">
                <div className="text-sm font-semibold tracking-tight whitespace-nowrap" style={{ color: 'var(--text-primary)' }}>Raptor</div>
                <div className="text-[10px] mt-0.5 whitespace-nowrap" style={{ color: 'var(--text-muted)' }}>Platform v0.3</div>
              </div>
            </>
          )}
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
          {!collapsed && (
            <p className="text-[10px] font-semibold uppercase tracking-widest px-3 mb-2 mt-1"
              style={{ color: 'var(--text-muted)' }}>
              Navigation
            </p>
          )}
          {NAV_ITEMS.map(item => (
            <NavItem key={item.id} item={item} active={activeTab === item.id}
              collapsed={collapsed} onClick={() => setActiveTab(item.id)} />
          ))}
        </nav>

        {/* Bottom */}
        <div className="px-2 pb-4 space-y-3" style={{ borderTop: '1px solid var(--border)' }}>

          {/* Theme switcher */}
          <div className="px-1 pt-3">
            {!collapsed && (
              <p className="text-[10px] font-semibold uppercase tracking-widest px-2 mb-2"
                style={{ color: 'var(--text-muted)' }}>
                Theme
              </p>
            )}
            <div className={`flex gap-2 ${collapsed ? 'flex-col items-center' : 'px-2'}`}>
              {Object.values(THEMES).map(t => (
                <button
                  key={t.key}
                  onClick={() => setThemeKey(t.key)}
                  title={t.label}
                  className="relative transition-transform duration-150 hover:scale-110 flex-shrink-0"
                  style={{
                    width: 18, height: 18,
                    borderRadius: '50%',
                    background: t.swatch,
                    boxShadow: themeKey === t.key
                      ? `0 0 0 2px var(--bg-surface), 0 0 0 4px var(--accent)`
                      : '0 0 0 1px rgba(255,255,255,0.1)',
                  }}
                />
              ))}
            </div>
          </div>

          {/* User card */}
          {!collapsed ? (
            <div className="rounded-xl px-3 py-2.5"
              style={{ background: 'var(--bg-raised)', border: '1px solid var(--border)' }}>
              <div className="flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                  style={{ background: 'linear-gradient(135deg,var(--accent),#6366f1)' }}>
                  {initials}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium truncate" style={{ color: 'var(--text-primary)' }}>{displayName}</div>
                  <div className="text-[10px] truncate" style={{ color: 'var(--text-muted)' }}>{user?.email || ''}</div>
                </div>
              </div>
              <LogoutButton onClick={handleLogout} />
            </div>
          ) : (
            <button
              onClick={handleLogout}
              title="Sign Out"
              className="w-full flex items-center justify-center py-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--text-muted)' }}
              onMouseEnter={e => e.currentTarget.style.color = '#f87171'}
              onMouseLeave={e => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-4 h-4">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 11l3-3m0 0l-3-3m3 3H6m0-7H4a1 1 0 00-1 1v10a1 1 0 001 1h2" />
              </svg>
            </button>
          )}

        </div>
      </aside>

      {/* ── Main ── */}
      <main className="flex-1 min-h-0 flex flex-col overflow-hidden" style={{ background: 'var(--bg-base)' }}>
        {children}
      </main>
    </div>
  )
}

function NavItem({ item, active, collapsed, onClick }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      title={collapsed ? item.label : undefined}
      className="w-full flex items-center gap-2.5 rounded-xl text-[13px] font-medium transition-all duration-150 relative"
      style={{
        padding: collapsed ? '10px 0' : '10px 12px',
        justifyContent: collapsed ? 'center' : 'flex-start',
        ...(active ? {
          background: 'var(--accent-dim)',
          color: 'var(--accent-text)',
          boxShadow: collapsed ? undefined : `inset 3px 0 0 var(--accent)`,
        } : hovered ? {
          background: 'var(--bg-overlay)',
          color: 'var(--text-primary)',
        } : {
          color: 'var(--text-muted)',
        })
      }}
    >
      <span className="flex-shrink-0 w-[17px] flex items-center justify-center">{item.icon}</span>
      {!collapsed && <span>{item.label}</span>}
      {!collapsed && active && (
        <span className="ml-auto w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ background: 'var(--accent)', boxShadow: '0 0 6px var(--accent)' }} />
      )}
    </button>
  )
}

function LogoutButton({ onClick }) {
  const [hovered, setHovered] = useState(false)
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="w-full flex items-center gap-2 px-1 py-1.5 mt-2 rounded-lg text-xs transition-all duration-150"
      style={hovered ? {
        color: '#f87171',
        background: 'rgba(239,68,68,0.08)',
      } : { color: 'var(--text-muted)' }}
    >
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-3.5 h-3.5 flex-shrink-0">
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 11l3-3m0 0l-3-3m3 3H6m0-7H4a1 1 0 00-1 1v10a1 1 0 001 1h2" />
      </svg>
      Sign Out
    </button>
  )
}
