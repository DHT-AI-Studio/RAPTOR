import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => sessionStorage.getItem('raptor_token') || null)
  const [user, setUser] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('raptor_user') || 'null') } catch { return null }
  })
  const [sessionExpired, setSessionExpired] = useState(false)

  const login = useCallback((accessToken, userInfo) => {
    sessionStorage.setItem('raptor_token', accessToken)
    sessionStorage.setItem('raptor_user', JSON.stringify(userInfo))
    setToken(accessToken)
    setUser(userInfo)
    setSessionExpired(false)
  }, [])

  const logout = useCallback(() => {
    sessionStorage.removeItem('raptor_token')
    sessionStorage.removeItem('raptor_user')
    setToken(null)
    setUser(null)
  }, [])

  useEffect(() => {
    const handler = () => {
      logout()
      setSessionExpired(true)
    }
    window.addEventListener('raptor:unauthorized', handler)
    return () => window.removeEventListener('raptor:unauthorized', handler)
  }, [logout])

  return (
    <AuthContext.Provider value={{ token, user, login, logout, isAuthenticated: !!token, sessionExpired }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
