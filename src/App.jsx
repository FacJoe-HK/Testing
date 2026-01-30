import { useEffect, useMemo, useState } from 'react'
import './App.css'

const timezones = [
  { label: 'Hong Kong', value: 'Asia/Hong_Kong' },
  { label: 'UTC', value: 'UTC' },
  { label: 'London', value: 'Europe/London' },
  { label: 'New York', value: 'America/New_York' },
  { label: 'Tokyo', value: 'Asia/Tokyo' },
]

function formatPct(n) {
  if (n === undefined || n === null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(1)}%`
}

function StrategyCard({ ticker, data }) {
  const best = Object.values(data.strategies).sort((a, b) => b.sharpe - a.sharpe)[0]
  return (
    <div className="mini-card">
      <div className="mini-header">
        <p className="eyebrow">{ticker}</p>
        <p className="mini-sub">Ensemble Sharpe {data.ensemble.sharpe.toFixed(2)}</p>
      </div>
      <div className="mini-row">
        <span>Best Sharpe</span>
        <strong>{best.name}</strong>
      </div>
      <div className="mini-grid">
        <div>
          <p className="label">Win rate</p>
          <p className="value">{formatPct(best.win_rate)}</p>
        </div>
        <div>
          <p className="label">Profit factor</p>
          <p className="value">{best.profit_factor === Infinity ? '∞' : best.profit_factor.toFixed(2)}</p>
        </div>
        <div>
          <p className="label">Sharpe</p>
          <p className="value">{best.sharpe.toFixed(2)}</p>
        </div>
        <div>
          <p className="label">Max DD</p>
          <p className="value">{formatPct(best.max_drawdown)}</p>
        </div>
        <div>
          <p className="label">Total return</p>
          <p className="value">{formatPct(best.total_return)}</p>
        </div>
        <div>
          <p className="label">Trades</p>
          <p className="value">{best.trades}</p>
        </div>
      </div>
    </div>
  )
}

function App() {
  const [now, setNow] = useState(new Date())
  const [is24h, setIs24h] = useState(true)
  const [timezone, setTimezone] = useState('Asia/Hong_Kong')
  const [bt, setBt] = useState(null)

  useEffect(() => {
    const tick = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(tick)
  }, [])

  useEffect(() => {
    fetch('/backtest.json')
      .then((res) => res.json())
      .then(setBt)
      .catch(() => {})
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

        <section className="card metrics-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Backtest · Stocks</p>
              <h2>Strategy metrics (PoC)</h2>
              <p className="subtitle">Universe: SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA.</p>
            </div>
            {!bt && <p className="subtitle">Loading...</p>}
          </div>
          {bt && (
            <div className="mini-grid-wrap">
              {Object.entries(bt.results).map(([ticker, data]) => (
                <StrategyCard key={ticker} ticker={ticker} data={data} />
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
