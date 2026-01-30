import { useEffect, useMemo, useState } from 'react'
import './App.css'

const timezones = [
  { label: 'Hong Kong', value: 'Asia/Hong_Kong' },
  { label: 'UTC', value: 'UTC' },
  { label: 'London', value: 'Europe/London' },
  { label: 'New York', value: 'America/New_York' },
  { label: 'Tokyo', value: 'Asia/Tokyo' },
]

function App() {
  const [now, setNow] = useState(new Date())
  const [is24h, setIs24h] = useState(true)
  const [timezone, setTimezone] = useState('Asia/Hong_Kong')

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  const timeString = useMemo(
    () =>
      new Intl.DateTimeFormat('en-US', {
        timeZone: timezone,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: !is24h,
      }).format(now),
    [now, timezone, is24h]
  )

  const dateString = useMemo(
    () =>
      new Intl.DateTimeFormat('en-US', {
        timeZone: timezone,
        weekday: 'long',
        month: 'long',
        day: 'numeric',
        year: 'numeric',
      }).format(now),
    [now, timezone]
  )

  const offsetLabel = useMemo(() => {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      timeZoneName: 'short',
    }).formatToParts(now)

    return parts.find((p) => p.type === 'timeZoneName')?.value ?? ''
  }, [now, timezone])

  return (
    <div className="app-shell">
      <div className="background-glow" aria-hidden="true" />
      <main className="layout">
        <header className="header">
          <div>
            <p className="eyebrow">Clocking · Modern</p>
            <h1>
              Minimal clock, <span className="accent">always-on</span> time.
            </h1>
            <p className="subtitle">
              Live time with timezone awareness. Built for quick glances and a
              clean desk setup.
            </p>
          </div>
          <div className="live-pill">
            <span className="dot" /> Live · {offsetLabel}
          </div>
        </header>

        <section className="grid">
          <div className="card clock-card">
            <div className="clock-ring" aria-hidden="true">
              <div className="clock-ring-inner" />
            </div>
            <div className="time-display">{timeString}</div>
            <div className="date-display">{dateString}</div>
            <div className="controls">
              <button
                className="pill"
                onClick={() => setIs24h((prev) => !prev)}
                aria-label="Toggle 12 or 24 hour format"
              >
                {is24h ? '24h' : '12h'}
              </button>
              <select
                className="select"
                value={timezone}
                onChange={(e) => setTimezone(e.target.value)}
                aria-label="Change timezone"
              >
                {timezones.map((zone) => (
                  <option key={zone.value} value={zone.value}>
                    {zone.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="card info-card">
            <div className="info-row">
              <p className="label">Current offset</p>
              <p className="value">{offsetLabel}</p>
            </div>
            <div className="info-row">
              <p className="label">Timezone</p>
              <p className="value">{timezones.find((t) => t.value === timezone)?.label}</p>
            </div>
            <div className="info-row">
              <p className="label">Format</p>
              <p className="value">{is24h ? '24-hour' : '12-hour'} mode</p>
            </div>
            <p className="tip">Tip: add this page as a PWA or keep it pinned for a desk-friendly clock.</p>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
