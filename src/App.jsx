import { useEffect, useMemo, useState } from 'react'
import './App.css'

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
        <p className="mini-sub">Ensemble Sharpe {data.ensemble?.sharpe?.toFixed(2) ?? '—'}</p>
      </div>
      <div className="mini-row">
        <span>Best Sharpe</span>
        <strong>{best?.name ?? '—'}</strong>
      </div>
      {best && (
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
      )}
    </div>
  )
}

function App() {
  const [bt, setBt] = useState(null)

  useEffect(() => {
    fetch('/backtest.json')
      .then((res) => res.json())
      .then(setBt)
      .catch(() => {})
  }, [])

  const offsetLabel = useMemo(() => {
    const now = new Date()
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Hong_Kong',
      timeZoneName: 'short',
    }).formatToParts(now)
    return parts.find((p) => p.type === 'timeZoneName')?.value ?? ''
  }, [])

  return (
    <div className="app-shell">
      <div className="background-glow" aria-hidden="true" />
      <main className="layout">
        <header className="header">
          <div>
            <p className="eyebrow">Backtest · Stocks</p>
            <h1>Strategy lab dashboard</h1>
            <p className="subtitle">Universe: SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOG, META, TSLA. Live sweeps.</p>
          </div>
          <div className="live-pill">
            <span className="dot" /> Live · {offsetLabel}
          </div>
        </header>

        <section className="card metrics-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Best Sharpe per ticker</p>
              <h2>Top configs by ticker</h2>
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
