import React, { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext(null)

export const THEMES = {
  midnight: {
    key: 'midnight',
    label: 'Midnight',
    swatch: 'linear-gradient(135deg,#1e3a5f,#0f172a)',
    vars: {
      '--bg-base':      '#0d192e',
      '--bg-surface':   '#122038',
      '--bg-raised':    '#182948',
      '--bg-overlay':   '#1f3358',
      '--border':       'rgba(255,255,255,0.08)',
      '--border-md':    'rgba(255,255,255,0.14)',
      '--border-lg':    'rgba(255,255,255,0.20)',
      '--accent':       '#3b82f6',
      '--accent-2':     '#2563eb',
      '--accent-dim':   'rgba(59,130,246,0.15)',
      '--accent-glow':  'rgba(59,130,246,0.28)',
      '--accent-text':  '#93c5fd',
      '--text-primary': '#ffffff',
      '--text-secondary':'#a0afc0',
      '--text-muted':   '#4e6480',
      '--glass-bg':     'rgba(13,25,46,0.80)',
      '--glass-border': 'rgba(255,255,255,0.07)',
      '--shadow-card':  '0 4px 24px rgba(0,0,0,0.45)',
      '--orb1':         'rgba(59,130,246,0.07)',
      '--orb2':         'rgba(99,102,241,0.05)',
      '--scheme':       'dark',
    },
  },
  light: {
    key: 'light',
    label: 'Light',
    swatch: 'linear-gradient(135deg,#e0eaff,#f0f4ff)',
    vars: {
      '--bg-base':      '#f3f6fd',
      '--bg-surface':   '#ffffff',
      '--bg-raised':    '#f8faff',
      '--bg-overlay':   '#eef2fb',
      '--border':       'rgba(0,0,0,0.07)',
      '--border-md':    'rgba(0,0,0,0.11)',
      '--border-lg':    'rgba(0,0,0,0.16)',
      '--accent':       '#2563eb',
      '--accent-2':     '#1d4ed8',
      '--accent-dim':   'rgba(37,99,235,0.10)',
      '--accent-glow':  'rgba(37,99,235,0.22)',
      '--accent-text':  '#1d4ed8',
      '--text-primary': '#0f172a',
      '--text-secondary':'#475569',
      '--text-muted':   '#64748b',
      '--glass-bg':     'rgba(255,255,255,0.80)',
      '--glass-border': 'rgba(0,0,0,0.07)',
      '--shadow-card':  '0 2px 12px rgba(0,0,0,0.08), 0 0 0 1px rgba(0,0,0,0.05)',
      '--orb1':         'rgba(37,99,235,0.06)',
      '--orb2':         'rgba(99,102,241,0.04)',
      '--scheme':       'light',
    },
  },
}

export function ThemeProvider({ children }) {
  const [themeKey, setThemeKey] = useState(
    () => localStorage.getItem('raptor_theme') || 'midnight'
  )

  const theme = THEMES[themeKey] || THEMES.midnight

  useEffect(() => {
    const root = document.documentElement
    Object.entries(theme.vars).forEach(([k, v]) => root.style.setProperty(k, v))
    root.setAttribute('data-theme', themeKey)
    root.style.colorScheme = theme.vars['--scheme']
    localStorage.setItem('raptor_theme', themeKey)
  }, [themeKey, theme])

  return (
    <ThemeContext.Provider value={{ themeKey, setThemeKey, theme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}
