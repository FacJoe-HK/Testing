import { useEffect, useMemo, useState } from 'react'
import './App.css'

function formatPct(n) {
  if (n === undefined || n === null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(1)}%`
}

function StrategyCard({ ticker, data }) {
  const entries = Object.entries(data.strategies || {})
  const bestTrain = entries
    .map(([name, obj]) => ({ name, ...obj.train, validation: obj.validation }))
    .sort((a, b) => b.sharpe - a.sharpe)[0]

  const topWin = entries
    .flatMap(([name, obj]) => (data.top_candidates?.[name] || []).map((c) => ({ ...c, strat: name })))
    .sort((a, b) => b.win_rate - a.win_rate)[0]

  return (
    <div className="mini-card">
      <div className="mini-header">
        <p className="eyebrow">{ticker}</p>
        <p className="mini-sub">Ensemble Sharpe {data.ensemble?.sharpe?.toFixed(2) ?? '—'}</p>
      </div>
      <div className="mini-row">
        <span>Best Sharpe (train)</span>
        <strong>{bestTrain?.name ?? '—'}</strong>
      </div>
      {bestTrain && (
        <div className="mini-grid">
          <div>
            <p className="label">Train win</p>
            <p className="value">{formatPct(bestTrain.win_rate)}</p>
          </div>
          <div>
            <p className="label">Train PF</p>
            <p className="value">{bestTrain.profit_factor === Infinity ? '∞' : bestTrain.profit_factor.toFixed(2)}</p>
          </div>
          <div>
            <p className="label">Train Sharpe</p>
            <p className="value">{bestTrain.sharpe.toFixed(2)}</p>
          </div>
          <div>
            <p className="label">Val win</p>
            <p className="value">{bestTrain.validation ? formatPct(bestTrain.validation.win_rate) : '—'}</p>
          </div>
          <div>
            <p className="label">Val PF</p>
            <p className="value">
              {bestTrain.validation
                ? bestTrain.validation.profit_factor === Infinity
                  ? '∞'
                  : bestTrain.validation.profit_factor.toFixed(2)
                : '—'}
            </p>
          </div>
          <div>
            <p className="label">Val Sharpe</p>
            <p className="value">{bestTrain.validation ? bestTrain.validation.sharpe.toFixed(2) : '—'}</p>
          </div>
        </div>
      )}
      {topWin && (
        <div className="mini-row">
          <span>Top win-rate (train)</span>
          <strong>
            {topWin.strat}: {formatPct(topWin.win_rate)} PF {topWin.profit_factor === Infinity ? '∞' : topWin.profit_factor.toFixed(2)}
          </strong>
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
