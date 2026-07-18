import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react'
import mpegts from 'mpegts.js'

function isTsSource(url, assetPath) {
  const check = s => {
    if (!s) return false
    const lower = s.toLowerCase()
    return lower.endsWith('.ts') || lower.endsWith('.m2ts') || lower.endsWith('.mts')
      || lower.includes('/mp2t/') || lower.includes('/mpeg-ts/')
  }
  return check(url?.split('?')[0]) || check(assetPath)
}

const VideoPlayer = forwardRef(function VideoPlayer({ src, assetPath, className, preload, onError }, ref) {
  const videoRef = useRef(null)
  const playerRef = useRef(null)
  const [transcodedUrl, setTranscodedUrl] = useState(null)
  const [converting, setConverting] = useState(false)
  // Refs so useImperativeHandle setters can read current state synchronously
  const convertingRef = useRef(false)
  const pendingSeekRef = useRef(null)
  const pendingPlayRef = useRef(false)

  const isTs = isTsSource(src, assetPath)

  useImperativeHandle(ref, () => ({
    get currentTime() { return videoRef.current?.currentTime ?? 0 },
    set currentTime(t) {
      if (!convertingRef.current && videoRef.current) {
        videoRef.current.currentTime = t
      } else {
        // Video is loading/transcoding — store and apply once ready
        pendingSeekRef.current = t
      }
    },
    play() {
      if (!convertingRef.current && videoRef.current) return videoRef.current.play()
      pendingPlayRef.current = true
      return Promise.resolve()
    },
    pause() { return videoRef.current?.pause() },
  }))

  useEffect(() => {
    if (!src || !videoRef.current || !isTs || !mpegts.isSupported()) return
    const player = mpegts.createPlayer({ type: 'mpegts', url: src })
    playerRef.current = player
    player.attachMediaElement(videoRef.current)
    player.on(mpegts.Events.ERROR, () => onError?.())
    player.load()
    return () => { player.destroy(); playerRef.current = null }
  }, [src, isTs])

  const triggerTranscode = () => {
    if (isTs || transcodedUrl || convertingRef.current) return
    convertingRef.current = true
    setConverting(true)
    setTranscodedUrl(`/transcode/?url=${encodeURIComponent(src)}`)
  }

  const handleNativeError = () => {
    if (isTs || transcodedUrl) return
    triggerTranscode()
  }

  // Black screen detection: H.265 silent fail on Linux Chrome
  const handleLoadedData = () => {
    if (isTs || transcodedUrl) return
    const v = videoRef.current
    if (v && v.videoHeight === 0 && v.duration > 0) triggerTranscode()
  }

  const handleCanPlay = () => {
    convertingRef.current = false
    setConverting(false)
    // Apply any seek/play that arrived while we were transcoding
    if (pendingSeekRef.current != null && videoRef.current) {
      videoRef.current.currentTime = pendingSeekRef.current
      pendingSeekRef.current = null
    }
    if (pendingPlayRef.current) {
      pendingPlayRef.current = false
      videoRef.current?.play().catch(() => {})
    }
  }

  const activeSrc = transcodedUrl || (isTs ? undefined : src)

  // The video element must stay in the DOM even while converting so it can
  // load the transcoded stream and fire onCanPlay. We hide it visually and
  // show a spinner on top; once onCanPlay fires we reveal the video.
  return (
    <>
      {converting && (
        <div className="w-full flex flex-col items-center justify-center gap-2 py-8 rounded-xl bg-black">
          <svg className="animate-spin h-6 w-6 text-blue-400" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 00-8 8h4z"/>
          </svg>
          <p className="text-xs text-gray-400">正在轉換 H.265 → H.264，請稍候…</p>
        </div>
      )}
      <video
        ref={videoRef}
        src={activeSrc}
        controls
        className={converting ? 'hidden' : className}
        preload={converting ? 'auto' : (preload ?? 'metadata')}
        onCanPlay={handleCanPlay}
        onError={isTs ? undefined : (transcodedUrl ? () => onError?.() : handleNativeError)}
        onLoadedData={isTs || transcodedUrl ? undefined : handleLoadedData}
      />
    </>
  )
})

export default VideoPlayer
