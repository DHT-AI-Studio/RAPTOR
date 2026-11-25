import React, { useEffect, useState } from 'react'
import { useAuth } from '../contexts/AuthContext.jsx'
import { apiGetConsent, apiSetConsent } from '../api/client.js'

const GENERAL_IDS = ['memory', 'analytics', 'benchmark']
const BIOMETRIC_IDS = ['voice_recording', 'face_detection', 'voice_clone']

const PURPOSE_LABELS = {
  memory: 'Conversation Memory',
  analytics: 'Usage Analytics',
  benchmark: 'Benchmark Data',
  voice_recording: 'Voice Recording',
  face_detection: 'Face Detection',
  voice_clone: 'Voice Cloning',
}

// Only purposes with a real deletion cascade on withdrawal need the confirm
// step -- matches the backend's own withdrawal_store mapping (analytics has
// none, its data is already aggregate).
const WITHDRAWAL_CONSEQUENCES = {
  memory: 'Your stored conversation history will be permanently deleted.',
  benchmark: 'Your saved benchmark schemas and results will be permanently deleted.',
  voice_recording: 'Your recorded voice sessions will be permanently deleted.',
  face_detection: 'Your face data will be permanently deleted: face groups, named people, and People Search results for your videos will stop working.',
  voice_clone: 'Your cloned-voice data will be permanently deleted, and voice cloning will stop working until you turn this back on.',
}

function formatDate(iso) {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
  } catch {
    return iso
  }
}

function ShieldIcon({ className }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.6" className={className}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M10 2.3l6.3 2.4v4.3c0 4.2-2.7 7-6.3 8.5-3.6-1.5-6.3-4.3-6.3-8.5V4.7z" />
    </svg>
  )
}

function Switch({ checked, disabled, onClick, label }) {
  return (
    <button
      type="button" role="switch" aria-checked={checked} aria-label={label}
      onClick={onClick} disabled={disabled}
      className="relative inline-flex items-center flex-shrink-0 rounded-full transition-colors
                 duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2
                 disabled:opacity-40 disabled:cursor-not-allowed"
      style={{
        width: 40, height: 22,
        background: checked ? 'var(--accent)' : 'var(--bg-overlay)',
        border: `1px solid ${checked ? 'var(--accent-2)' : 'var(--border-md)'}`,
        outlineColor: 'var(--accent)',
      }}
    >
      <span
        className="absolute rounded-full transition-transform duration-150"
        style={{
          width: 16, height: 16, background: '#fff',
          boxShadow: '0 1px 2px rgba(0,0,0,0.35)',
          transform: checked ? 'translateX(20px)' : 'translateX(2px)',
        }}
      />
    </button>
  )
}

function ConsentRow({ record, pending, onToggle }) {
  const { id, granted, description, granted_at } = record
  return (
    <div className="flex items-start justify-between gap-4 py-3" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="min-w-0">
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
          {PURPOSE_LABELS[id] || id}
        </p>
        <p className="text-xs mt-0.5" style={{ color: 'var(--text-secondary)' }}>{description}</p>
        <p className="text-[11px] mt-1" style={{ color: 'var(--text-muted)' }}>
          {granted ? `Granted${granted_at ? ' on ' + formatDate(granted_at) : ''}` : 'Not granted'}
        </p>
      </div>
      <Switch checked={granted} disabled={pending} onClick={() => onToggle(id, granted)}
        label={PURPOSE_LABELS[id] || id} />
    </div>
  )
}

function ConfirmWithdrawModal({ purposeId, busy, onConfirm, onCancel }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.55)' }} onClick={onCancel}>
      <div className="card max-w-sm w-full" style={{ background: 'var(--bg-surface)' }}
        onClick={e => e.stopPropagation()}>
        <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
          Turn off {PURPOSE_LABELS[purposeId] || purposeId}?
        </h3>
        <p className="text-sm mb-5" style={{ color: 'var(--text-secondary)' }}>
          {WITHDRAWAL_CONSEQUENCES[purposeId]}
        </p>
        <div className="flex justify-end gap-2">
          <button className="btn-secondary" onClick={onCancel} disabled={busy}>Cancel</button>
          <button className="btn-primary" style={{ background: '#ef4444' }} onClick={onConfirm} disabled={busy}>
            {busy ? 'Turning off…' : 'Turn off and delete'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function PrivacyPanel() {
  const { token } = useAuth()
  const [purposes, setPurposes] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [pendingId, setPendingId] = useState(null)
  const [confirmId, setConfirmId] = useState(null)

  const load = () => {
    setLoading(true); setError('')
    apiGetConsent(token)
      .then(data => setPurposes(data.purposes))
      .catch(err => setError(err.message || 'Could not load your privacy settings.'))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const applyChange = async (id, granted) => {
    setPendingId(id); setError('')
    try {
      await apiSetConsent(token, id, granted)
      const data = await apiGetConsent(token)
      setPurposes(data.purposes)
    } catch (err) {
      setError(err.message || `Could not update ${PURPOSE_LABELS[id] || id}.`)
    } finally {
      setPendingId(null); setConfirmId(null)
    }
  }

  const handleToggle = (id, currentlyGranted) => {
    if (currentlyGranted && WITHDRAWAL_CONSEQUENCES[id]) {
      setConfirmId(id)
      return
    }
    applyChange(id, !currentlyGranted)
  }

  const byId = Object.fromEntries((purposes || []).map(p => [p.id, p]))

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="page-header">
        <h1 className="page-title">Privacy</h1>
        <p className="page-desc">Control what Raptor is allowed to do with your data.</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {error && <div className="error-box mb-4">{error}</div>}

        {loading && !purposes ? (
          <div className="flex flex-col items-center justify-center py-20">
            <svg className="animate-spin h-8 w-8 mb-4" viewBox="0 0 24 24" fill="none" style={{ color: 'var(--accent)' }}>
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z" />
            </svg>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading your privacy settings…</p>
          </div>
        ) : !purposes ? null : (
          <div className="max-w-2xl space-y-4">
            <div className="card">
              <p className="text-[11px] font-semibold uppercase tracking-widest mb-1"
                style={{ color: 'var(--text-muted)' }}>
                General
              </p>
              {GENERAL_IDS.filter(id => byId[id]).map(id => (
                <ConsentRow key={id} record={byId[id]} pending={pendingId === id} onToggle={handleToggle} />
              ))}
            </div>

            <div className="card">
              <div className="flex items-center gap-1.5 mb-1">
                <ShieldIcon className="w-3.5 h-3.5" />
                <p className="text-[11px] font-semibold uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                  Biometric Data
                </p>
              </div>
              <p className="text-xs mb-1" style={{ color: 'var(--text-secondary)' }}>
                These identify you personally and fall under GDPR Article 9. They stay off until you turn them on.
              </p>
              {BIOMETRIC_IDS.filter(id => byId[id]).map(id => (
                <ConsentRow key={id} record={byId[id]} pending={pendingId === id} onToggle={handleToggle} />
              ))}
            </div>
          </div>
        )}
      </div>

      {confirmId && (
        <ConfirmWithdrawModal
          purposeId={confirmId}
          busy={pendingId === confirmId}
          onConfirm={() => applyChange(confirmId, false)}
          onCancel={() => setConfirmId(null)}
        />
      )}
    </div>
  )
}
