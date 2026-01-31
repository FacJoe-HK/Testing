import { useEffect, useMemo, useState } from 'react'
import './App.css'

function formatPct(n) {
  if (n === undefined || n === null || Number.isNaN(n)) return '—'
  return `${(n * 100).toFixed(1)}%`
}

function formatPf(pf) {
  if (pf === undefined || pf === null || Number.isNaN(pf)) return '—'
  if (pf === Infinity) return '∞'
  return pf.toFixed(2)
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
            <p className="value">{formatPf(bestTrain.profit_factor)}</p>
          </div>
          <div>
            <p className="label">Train Sharpe</p>
            <p className="value">{bestTrain.sharpe?.toFixed(2)}</p>
          </div>
          <div>
            <p className="label">Val win</p>
            <p className="value">{bestTrain.validation ? formatPct(bestTrain.validation.win_rate) : '—'}</p>
          </div>
          <div>
            <p className="label">Val PF</p>
            <p className="value">{bestTrain.validation ? formatPf(bestTrain.validation.profit_factor) : '—'}</p>
          </div>
          <div>
            <p className="label">Val Sharpe</p>
            <p className="value">{bestTrain.validation ? bestTrain.validation.sharpe?.toFixed(2) : '—'}</p>
          </div>
        </div>
      )}
      {topWin && (
        <div className="mini-row">
          <span>Top win-rate (train)</span>
          <strong>
            {topWin.strat}: {formatPct(topWin.win_rate)} PF {formatPf(topWin.profit_factor)}
          </strong>
        </div>
      )}
    </div>
  )
}

function TableRow({ ticker, strat, train, validation }) {
  return (
    <div className="table-row">
      <div className="cell ticker">{ticker}</div>
      <div className="cell">{strat}</div>
      <div className="cell">{formatPct(train?.win_rate)}</div>
      <div className="cell">{formatPf(train?.profit_factor)}</div>
      <div className="cell">{formatPct(train?.max_drawdown)}</div>
      <div className="cell">{train?.trades ?? '—'}</div>
      <div className="cell">{formatPct(validation?.win_rate)}</div>
      <div className="cell">{formatPf(validation?.profit_factor)}</div>
      <div className="cell">{formatPct(validation?.max_drawdown)}</div>
      <div className="cell">{validation?.trades ?? '—'}</div>
    </div>
  )
}

function App() {
  const [bt, setBt] = useState(null)
  const [lastFetch, setLastFetch] = useState(null)

  const load = () => {
    fetch('/backtest.json')
      .then((res) => res.json())
      .then((data) => {
        setBt(data)
        setLastFetch(new Date())
      })
      .catch(() => {})
  }

  useEffect(() => {
    load()
  }, [])

  const offsetLabel = useMemo(() => {
    const now = new Date()
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: 'Asia/Hong_Kong',
      timeZoneName: 'short',
    }).formatToParts(now)
    return parts.find((p) => p.type === 'timeZoneName')?.value ?? ''
  }, [])

  const topWinTrain = useMemo(() => {
    if (!bt) return []
    const rows = []
    Object.entries(bt.results).forEach(([ticker, data]) => {
      Object.entries(data.strategies || {}).forEach(([name, obj]) => {
        if (obj.train) rows.push({ ticker, strat: name, ...obj.train })
      })
    })
    return rows.sort((a, b) => b.win_rate - a.win_rate).slice(0, 5)
  }, [bt])

  const topWinVal = useMemo(() => {
    if (!bt) return []
    const rows = []
    Object.entries(bt.results).forEach(([ticker, data]) => {
      Object.entries(data.strategies || {}).forEach(([name, obj]) => {
        if (obj.validation) rows.push({ ticker, strat: name, ...obj.validation })
      })
    })
    return rows.sort((a, b) => b.win_rate - a.win_rate).slice(0, 5)
  }, [bt])

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

        <section className="card controls-card">
          <div className="controls-row">
            <div>
              <p className="label">Data</p>
              <p className="value">from backtest.json (train/validation split)</p>
            </div>
            <div className="controls-buttons">
              <button className="pill" onClick={load}>Refresh data</button>
              <p className="mini-sub">{lastFetch ? `Last fetch: ${lastFetch.toLocaleTimeString()}` : 'Not loaded'}</p>
            </div>
          </div>
        </section>

        <section className="card metrics-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Per ticker</p>
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

        <section className="card table-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Top win-rate (train)</p>
              <h2>Conservative focus</h2>
            </div>
          </div>
          <div className="table">
            <div className="table-row head">
              <div className="cell ticker">Ticker</div>
              <div className="cell">Strat</div>
              <div className="cell">Train win</div>
              <div className="cell">Train PF</div>
              <div className="cell">Train DD</div>
              <div className="cell">Trades</div>
              <div className="cell">Val win</div>
              <div className="cell">Val PF</div>
              <div className="cell">Val DD</div>
              <div className="cell">Trades</div>
            </div>
            {bt &&
              Object.entries(bt.results).map(([ticker, data]) =>
                Object.entries(data.strategies || {}).map(([name, obj]) => (
                  <TableRow
                    key={`${ticker}-${name}`}
                    ticker={ticker}
                    strat={name}
                    train={obj.train}
                    validation={obj.validation}
                  />
                ))
              )}
          </div>
        </section>

        <section className="card table-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Top win-rate (train)</p>
              <h2>Top 5 train</h2>
            </div>
          </div>
          <div className="table">
            <div className="table-row head">
              <div className="cell ticker">Ticker</div>
              <div className="cell">Strat</div>
              <div className="cell">Win</div>
              <div className="cell">PF</div>
              <div className="cell">DD</div>
              <div className="cell">Trades</div>
            </div>
            {topWinTrain.map((r) => (
              <div className="table-row" key={`${r.ticker}-${r.strat}-train`}>
                <div className="cell ticker">{r.ticker}</div>
                <div className="cell">{r.strat}</div>
                <div className="cell">{formatPct(r.win_rate)}</div>
                <div className="cell">{formatPf(r.profit_factor)}</div>
                <div className="cell">{formatPct(r.max_drawdown)}</div>
                <div className="cell">{r.trades ?? '—'}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="card table-card">
          <div className="metrics-header">
            <div>
              <p className="eyebrow">Top win-rate (validation)</p>
              <h2>Top 5 validation</h2>
            </div>
          </div>
          <div className="table">
            <div className="table-row head">
              <div className="cell ticker">Ticker</div>
              <div className="cell">Strat</div>
              <div className="cell">Win</div>
              <div className="cell">PF</div>
              <div className="cell">DD</div>
              <div className="cell">Trades</div>
            </div>
            {topWinVal.map((r) => (
              <div className="table-row" key={`${r.ticker}-${r.strat}-val`}>
                <div className="cell ticker">{r.ticker}</div>
                <div className="cell">{r.strat}</div>
                <div className="cell">{formatPct(r.win_rate)}</div>
                <div className="cell">{formatPf(r.profit_factor)}</div>
                <div className="cell">{formatPct(r.max_drawdown)}</div>
                <div className="cell">{r.trades ?? '—'}</div>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
