"""Halal opportunity scanner (runs on GitHub Actions, free, no AI).

Full scan  : python scan.py              -> docs/data.json (+ GitHub issue when NEW buy signals appear)
One ticker : TICKER=ADBE python scan.py  -> adds a detailed check of that stock to docs/data.json

Order of work (agreed with Hussein):
  1. universe  : all US-listed stocks with market cap > $2B (daily list from github.com/rreichel3/US-Stock-Symbols)
  2. strategies: (1) pullback to the respected moving average in an uptrend, (2) PEAD = buy the first dip after a
                 strong earnings gap, (3) quality company on sale (20-40% below its high, fundamentals intact)
  3. shariah   : only on the stocks that passed a strategy -> excluded industries, debt < 30% and cash < 30% of
                 market cap (AAOIFI ratios), Zoya "not compliant" list. In HLAL/SPUS = confirmed, otherwise = needs
                 confirmation before buying. Purification is accepted.
Educational tool, not financial advice.
"""
import json, os, re, subprocess, sys, time, datetime as dt
import numpy as np, pandas as pd, requests, yfinance as yf

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, DOCS = f"{ROOT}/data", f"{ROOT}/docs"
C = dict(
    min_mcap=2e9, min_price=5.0, min_dollar_vol=10e6,          # universe
    mas=[20, 50, 100, 150, 200], touch_pct=0.01, rising_lb=20, cooldown=15,
    bounce_pct=0.05, bounce_days=20, break_pct=0.03, oos_bars=504,   # respected-MA test (5 years, last 2 = check)
    zone_above=0.02, zone_below=0.03, near_pct=0.05,
    mom_top=0.70,                                                # pullback: top 30% 12-1 momentum
    gap_min=0.05, gap_vol=2.0, pead_min_days=2, pead_max_days=20, pead_dip=0.03,
    sale_min=0.20, sale_max=0.40,
    earn_block_days=5, max_debt=0.30, max_cash=0.30, max_results=12,
)
BAD_INDUSTRY = ["bank", "insurance", "credit services", "capital markets", "mortgage", "financial conglomerate",
                "asset management", "brewer", "winer", "distiller", "tobacco", "gambling", "casino", "reit - mortgage",
                "financial data", "shell companies"]
STRAT = {"pullback": "ارتداد في اتجاه صاعد", "pead": "ما بعد الأرباح القوية", "sale": "جودة بسعر مخفّض"}
SRC = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/{ex}/{ex}_full_tickers.json"
now = dt.datetime.now(dt.timezone.utc)


def log(*a): print(*a, flush=True)
def jload(p, d):
    try: return json.load(open(p))
    except Exception: return d
def r2(x, n=2):
    try: return None if x is None or not np.isfinite(x) else round(float(x), n)
    except Exception: return None


# ---------------------------------------------------------------- 1. universe
def universe():
    rows = []
    try:
        for ex in ("nasdaq", "nyse", "amex"):
            r = requests.get(SRC.format(ex=ex), timeout=60); r.raise_for_status(); rows += r.json()
        json.dump(rows, open(f"{DATA}/universe_raw.json", "w"))
    except Exception as e:
        log("universe download failed, using cached list:", e); rows = jload(f"{DATA}/universe_raw.json", [])
    bad_name = re.compile(r"preferred|depositary shares,? each|notes due|warrant|\bunits?\b|\brights?\b|%|perpetual|debenture", re.I)
    out = {}
    for x in rows:
        s = (x.get("symbol") or "").strip()
        if not re.fullmatch(r"[A-Z]{1,5}(/[A-Z])?", s): continue
        try: mc = float(x.get("marketCap") or 0); px = float(str(x.get("lastsale", "0")).replace("$", "") or 0)
        except ValueError: continue
        if mc < C["min_mcap"] or px < C["min_price"] or bad_name.search(x.get("name", "")): continue
        if x.get("sector") == "Finance" and x.get("industry") != "Real Estate": continue   # conventional finance
        out[s.replace("/", "-")] = {"name": re.sub(r"\s+(Common Stock|Class [A-C] Common Stock|Ordinary Shares).*$", "", x.get("name", "")),
                                    "sector": x.get("sector", ""), "industry": x.get("industry", ""), "mcap": mc}
    return out


# ---------------------------------------------------------------- prices
def download(tickers, period="5y"):
    frames = {}
    for i in range(0, len(tickers), 150):
        chunk = tickers[i:i + 150]
        for a in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                                 threads=True, progress=False)
                break
            except Exception as e:
                log("download retry", a, e); time.sleep(10 * (a + 1)); df = None
        if df is None: continue
        for t in chunk:
            try:
                d = df[t] if len(chunk) > 1 else df
                d = d.dropna(subset=["Close"])
                if len(d) >= 260: frames[t] = d
            except Exception:
                pass
        log(f"prices {min(i + 150, len(tickers))}/{len(tickers)}")
    return frames


def sma(a, n):
    out = np.full(len(a), np.nan); c = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


# ---------------------------------------------------------------- strategy 1: respected moving average
def ma_events(close, low, ma, start, end):
    ev, last = [], -10**9
    lb, bd = C["rising_lb"], C["bounce_days"]
    for i in range(max(start, lb + 1), min(end, len(close) - bd - 1)):
        m, mp, mr = ma[i], ma[i - 1], ma[i - lb]
        if np.isnan(m) or np.isnan(mp) or np.isnan(mr) or i - last <= C["cooldown"]: continue
        if m > mr and close[i - 1] > mp * (1 + C["touch_pct"]) and low[i] <= m * (1 + C["touch_pct"]):
            ok = False
            for j in range(i + 1, i + 1 + bd):
                if close[j] < ma[j] * (1 - C["break_pct"]): break
                if close[j] >= close[i] * (1 + C["bounce_pct"]): ok = True; break
            ev.append((ok, i)); last = i
    return ev


def respected_ma(d):
    close, low = d["Close"].to_numpy(float), d["Low"].to_numpy(float)
    n, split = len(close), len(close) - C["oos_bars"]
    best = None
    for p in C["mas"]:
        ma = sma(close, p)
        ev = ma_events(close, low, ma, 0, n)
        tot = len(ev); b = sum(ok for ok, _ in ev)
        oos = [ok for ok, i in ev if i >= split]
        r = dict(ma=p, value=ma[-1], touches=tot, bounces=b, rate=b / tot if tot else 0, smoothed=(b + 1) / (tot + 2),
                 oos_touches=len(oos), oos_rate=sum(oos) / len(oos) if oos else 0, rising=ma[-1] > ma[-1 - C["rising_lb"]],
                 line=ma)
        if tot >= 3 and (best is None or r["smoothed"] > best["smoothed"]): best = r
    if not best: return None
    if best["touches"] >= 5 and best["rate"] >= 0.5 and best["oos_touches"] >= 1 and best["oos_rate"] >= 0.5: best["conf"] = "عالية"
    elif best["rate"] >= 0.4: best["conf"] = "متوسطة"
    else: best["conf"] = "ضعيفة"
    return best


def zone_status(price, lo, hi):
    if lo <= price <= hi: return "in"
    if price > hi and price / hi - 1 <= C["near_pct"]: return "near"
    return None


def strat_pullback(t, d, f):
    if not (f["above200"] and f["rising200"] and f["mom_pct"] >= C["mom_top"]): return None
    m = respected_ma(d)
    if not m or m["conf"] == "ضعيفة" or not m["rising"]: return None
    lo, hi = m["value"] * (1 - C["zone_below"]), m["value"] * (1 + C["zone_above"])
    st = zone_status(f["price"], lo, hi)
    if not st: return None
    score = 45 * f["mom_pct"] + 35 * m["smoothed"] + (20 if st == "in" else 10)
    why = (f"اتجاه صاعد قوي (أداء ١٢ شهرًا أفضل من {round(f['mom_pct']*100)}٪ من السوق)، "
           f"ونزل إلى متوسط {m['ma']} يوم الذي ارتدّ منه {m['bounces']} من {m['touches']} مرة خلال ٥ سنوات (ثقة {m['conf']}).")
    return dict(strategy="pullback", status=st, zone_low=lo, zone_high=hi, score=score, why=why,
                ma=m["ma"], ma_line=m["line"], conf=m["conf"], respect=f"{m['bounces']}/{m['touches']}")


# ---------------------------------------------------------------- strategy 2: PEAD
def strat_pead(t, d, f):
    o, h, l, c, v = (d[k].to_numpy(float) for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(c); avgv = pd.Series(v).rolling(50).mean().shift(1).to_numpy()
    for g in range(n - 1 - C["pead_min_days"], n - 2 - C["pead_max_days"], -1):
        jump = c[g] / c[g - 1] - 1
        if jump >= C["gap_min"] and o[g] > c[g - 1] * 1.02 and avgv[g] > 0 and v[g] >= C["gap_vol"] * avgv[g]:
            break
    else:
        return None
    hi_after = h[g:].max(); price = c[-1]
    if price > hi_after * (1 - C["pead_dip"]) or price < l[g] or price < f["sma50"]: return None
    lo, hi = l[g], max(c[g], l[g] * 1.02)
    st = zone_status(price, lo, hi)
    if not st: return None
    days = n - 1 - g
    score = 40 + min(25, jump * 150) + min(15, (v[g] / avgv[g]) * 3) + (20 if st == "in" else 10)
    why = (f"قفز {round(jump*100)}٪ قبل {days} يوم تداول بحجم تداول {round(v[g]/avgv[g],1)}× المعتاد، "
           f"وتراجع الآن {round((1-price/hi_after)*100)}٪ عن القمة بعد القفزة مع بقائه فوق قاع يوم القفزة.")
    return dict(strategy="pead", status=st, zone_low=lo, zone_high=hi, score=score, why=why, gap_date=str(d.index[g].date()),
                gap_pct=round(jump * 100, 1))


# ---------------------------------------------------------------- strategy 3: quality on sale (price part; fundamentals later)
def strat_sale(t, d, f):
    if not (C["sale_min"] <= f["drawdown"] <= C["sale_max"]): return None
    c = d["Close"].to_numpy(float)
    if len(c) < 760 or c[-1] <= c[-756]: return None          # long-term winner: higher than 3 years ago
    if c[-1] < f["sma20"]: return None                          # stopped falling: back above the 20-day average
    lo, hi = f["sma20"] * 0.97, f["sma20"] * 1.03
    st = zone_status(f["price"], lo, hi) or "near"
    score = 40 + f["drawdown"] * 60
    why = f"أقل من قمته السنوية بـ {round(f['drawdown']*100)}٪ وبدأ يستقر فوق متوسط ٢٠ يوم، وسعره أعلى مما كان قبل ٣ سنوات."
    return dict(strategy="sale", status=st, zone_low=lo, zone_high=hi, score=score, why=why)


# ---------------------------------------------------------------- features
def features(d):
    c = d["Close"].to_numpy(float); v = d["Volume"].to_numpy(float)
    s200, s50, s20 = sma(c, 200), sma(c, 50), sma(c, 20)
    return dict(price=c[-1], above200=c[-1] > s200[-1], rising200=s200[-1] > s200[-21], sma50=s50[-1], sma20=s20[-1],
                mom=(c[-22] / c[-253] - 1) if len(c) > 253 else np.nan, drawdown=1 - c[-1] / c[-252:].max(),
                dollar_vol=float(np.nanmean(c[-50:] * v[-50:])), chg1d=c[-1] / c[-2] - 1)


# ---------------------------------------------------------------- 3. shariah + fundamentals (candidates only)
ETF = jload(f"{DATA}/halal_etf.json", {})
ETF_SET = set(ETF.get("HLAL", [])) | set(ETF.get("SPUS", []))
ZOYA_BLOCK = {x.strip() for x in open(f"{DATA}/zoya_block.txt").read().split() if x.strip()} if os.path.exists(f"{DATA}/zoya_block.txt") else set()


def info_of(t):
    for a in range(3):
        try:
            i = yf.Ticker(t).info or {}
            if i.get("marketCap") or i.get("industry"): return i
        except Exception as e:
            log("info retry", t, e)
        time.sleep(3 * (a + 1))
    return {}


_fx = {}
def to_usd(cur):
    """rate to convert a balance-sheet currency to USD (ADRs report in TWD, EUR, ...)."""
    if not cur or cur == "USD": return 1.0
    if cur not in _fx:
        try: _fx[cur] = float(yf.download(f"{cur}USD=X", period="5d", progress=False, auto_adjust=True)["Close"].dropna().iloc[-1].squeeze())
        except Exception: _fx[cur] = None
    return _fx[cur]


def shariah(t, i):
    """returns (badge, note): ✅ confirmed / 🟡 needs confirmation / ❌ not compliant"""
    if t in ZOYA_BLOCK: return "❌", "Zoya: غير متوافق"
    ind = (i.get("industry") or "").lower()
    if any(b in ind for b in BAD_INDUSTRY): return "❌", f"قطاع غير متوافق ({i.get('industry')})"
    mc = i.get("marketCap") or 0
    debt, cash = i.get("totalDebt"), i.get("totalCash")
    fx = to_usd(i.get("financialCurrency"))
    if fx is None: debt = cash = None
    else: debt, cash = (debt * fx if debt is not None else None), (cash * fx if cash is not None else None)
    if mc and debt is not None and debt / mc >= C["max_debt"]: return "❌", f"الديون {round(debt/mc*100)}٪ من القيمة السوقية (الحد ٣٠٪)"
    if mc and cash is not None and cash / mc >= C["max_cash"]: return "❌", f"النقد {round(cash/mc*100)}٪ من القيمة السوقية (الحد ٣٠٪)"
    ratios = f"الديون {round(debt/mc*100)}٪ · النقد {round(cash/mc*100)}٪" if mc and debt is not None and cash is not None else "بيانات الديون ناقصة"
    if t in ETF_SET: return "✅", f"ضمن صناديق الشريعة (HLAL/SPUS) · {ratios}"
    return "🟡", f"نجح بالفحص الأولي ({ratios}) — أكّد نسبة الإيرادات المحرّمة قبل الشراء"


def quality_ok(i):
    om, roe, fcf, rg = i.get("operatingMargins"), i.get("returnOnEquity"), i.get("freeCashflow"), i.get("revenueGrowth")
    fpe, tpe = i.get("forwardPE"), i.get("trailingPE")
    good = ((om or 0) >= 0.15 or (roe or 0) >= 0.15) and (fcf or 0) > 0 and (rg if rg is not None else -1) > 0
    cheap = fpe is not None and 0 < fpe < 35 and (tpe is None or fpe <= tpe * 1.05)
    return good and cheap


def pead_confirmed(t, gap_date):
    """latest reported quarter beat estimates and the gap happened 2-12 weeks after that quarter ended."""
    try:
        eh = yf.Ticker(t).get_earnings_history()
        if eh is None or eh.empty: return None
        q = pd.Timestamp(eh.index[-1]); sp = float(eh["surprisePercent"].iloc[-1])
        gd = pd.Timestamp(gap_date)
        if 10 <= (gd - q).days <= 90: return sp
        return None
    except Exception:
        return None


def next_earn_days(i):
    ts = i.get("earningsTimestampStart") or i.get("earningsTimestamp")
    if not ts: return None
    days = (dt.datetime.fromtimestamp(ts, dt.timezone.utc).date() - now.date()).days
    return days if days >= 0 else None


def enrich(t, sig, meta, d, i, eurusd):
    price = float(d["Close"].iloc[-1])
    badge, note = shariah(t, i)
    ed = next_earn_days(i)
    action = "اشترِ الدفعة الأولى" if sig["status"] == "in" else f"جهّز أمر محدد عند {sig['zone_high']:.2f}$"
    warn = []
    if ed is not None and ed <= C["earn_block_days"] and sig["strategy"] != "pead":
        action = "انتظر إعلان الأرباح"; warn.append(f"الأرباح بعد {ed} يوم")
    if badge == "🟡": warn.append("أكّد الشرعية قبل الشراء")
    tail = d["Close"].iloc[-130:]
    line = sig.get("ma_line")
    out = dict(ticker=t, name=i.get("shortName") or meta.get("name", t), sector=i.get("sector") or meta.get("sector", ""),
               strategy=sig["strategy"], strategy_ar=STRAT[sig["strategy"]], status=sig["status"], action=action,
               score=int(round(min(99, sig["score"]))), price=r2(price), price_eur=r2(price / eurusd) if eurusd else None,
               zone_low=r2(sig["zone_low"]), zone_high=r2(sig["zone_high"]),
               zone_low_eur=r2(sig["zone_low"] / eurusd) if eurusd else None, zone_high_eur=r2(sig["zone_high"] / eurusd) if eurusd else None,
               shariah=badge, shariah_note=note, why=sig["why"], warn=warn, earn_days=ed,
               chg1d=r2((d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100, 1),
               target=r2(i.get("targetMeanPrice")), analysts=i.get("numberOfAnalystOpinions"), rec=i.get("recommendationKey"),
               fpe=r2(i.get("forwardPE"), 1), margin=r2((i.get("operatingMargins") or 0) * 100, 1) if i.get("operatingMargins") is not None else None,
               growth=r2((i.get("revenueGrowth") or 0) * 100, 1) if i.get("revenueGrowth") is not None else None,
               spark=[r2(x) for x in tail.tolist()],
               spark_ma=[r2(x) for x in line[-130:].tolist()] if line is not None else None,
               extra={k: sig[k] for k in ("ma", "conf", "respect", "gap_date", "gap_pct", "surprise") if k in sig})
    return out


# ---------------------------------------------------------------- run
def market_box(frames_small):
    out = {}
    for t, lab in (("SPY", "S&P 500"), ("QQQ", "ناسداك 100"), ("HLAL", "صندوق HLAL الحلال")):
        d = frames_small.get(t)
        if d is None: continue
        c = d["Close"].to_numpy(float); s200 = sma(c, 200)
        out[t] = dict(label=lab, price=r2(c[-1]), chg1d=r2((c[-1] / c[-2] - 1) * 100, 1),
                      chg1m=r2((c[-1] / c[-22] - 1) * 100, 1), above200=bool(c[-1] > s200[-1]))
    return out


def full_scan():
    t0 = time.time()
    U = universe(); log("universe", len(U))
    if os.environ.get("LIMIT"): U = dict(sorted(U.items(), key=lambda x: -x[1]["mcap"])[:int(os.environ["LIMIT"])])
    frames = download(sorted(U))
    small = download(["SPY", "QQQ", "HLAL", "EURUSD=X"], period="2y")
    eurusd = float(small["EURUSD=X"]["Close"].iloc[-1]) if "EURUSD=X" in small else None
    F = {}
    for t, d in frames.items():
        try:
            f = features(d)
            if f["dollar_vol"] >= C["min_dollar_vol"] and np.isfinite(f["mom"]): F[t] = f
        except Exception:
            pass
    moms = pd.Series({t: f["mom"] for t, f in F.items()}).rank(pct=True)
    for t in F: F[t]["mom_pct"] = float(moms[t])
    log("liquid stocks", len(F))

    raw = []
    for t, f in F.items():
        d = frames[t]
        for fn in (strat_pullback, strat_pead, strat_sale):
            try:
                s = fn(t, d, f)
                if s: raw.append((t, s))
            except Exception as e:
                log("strategy error", t, fn.__name__, e)
    log("strategy candidates", len(raw), {k: sum(1 for _, s in raw if s["strategy"] == k) for k in STRAT})

    infos, signals, dropped = {}, [], []
    for t, s in sorted(raw, key=lambda x: -x[1]["score"]):
        if t not in infos: infos[t] = info_of(t); time.sleep(0.4)
        i = infos[t]
        if s["strategy"] == "sale" and not quality_ok(i):
            continue
        if s["strategy"] == "pead":
            sp = pead_confirmed(t, s["gap_date"])
            if sp is None or sp <= 0: continue
            s["surprise"] = round(sp * 100, 1); s["score"] += min(15, sp * 100)
            s["why"] += f" الأرباح فاقت التوقعات بـ {round(sp*100,1)}٪."
        row = enrich(t, s, U.get(t, {}), frames[t], i, eurusd)
        if row["shariah"] == "❌": dropped.append({"ticker": t, "strategy": s["strategy"], "reason": row["shariah_note"]}); continue
        signals.append(row)

    final = []
    for k in STRAT:
        rows = sorted([r for r in signals if r["strategy"] == k], key=lambda r: -r["score"])
        final += [r for r in rows if r["status"] == "in"][:8] + [r for r in rows if r["status"] != "in"][:6]

    prev = jload(f"{DATA}/state.json", {})
    prev_keys = set(prev.get("buy_keys", []))
    buy_keys = sorted({f'{r["ticker"]}|{r["strategy"]}' for r in final if r["status"] == "in" and r["action"].startswith("اشترِ")})
    all_keys = {f'{r["ticker"]}|{r["strategy"]}' for r in final}
    new = [r for r in final if f'{r["ticker"]}|{r["strategy"]}' in set(buy_keys) - prev_keys]
    gone = sorted(k for k in set(prev.get("all_keys", [])) - all_keys)
    for r in final: r["is_new"] = f'{r["ticker"]}|{r["strategy"]}' not in set(prev.get("all_keys", []))

    out = jload(f"{DOCS}/data.json", {})
    out.update(updated=now.isoformat(), eurusd=r2(eurusd, 4), market=market_box(small), signals=final,
               dropped_shariah=dropped[:40], gone=gone,
               stats=dict(universe=len(U), liquid=len(F), candidates=len(raw), shown=len(final), seconds=int(time.time() - t0)),
               config=dict(strategies=STRAT, etf_as_of=ETF.get("as_of")))
    json.dump(out, open(f"{DOCS}/data.json", "w"), ensure_ascii=False, separators=(",", ":"))
    json.dump({"buy_keys": buy_keys, "all_keys": sorted(all_keys), "date": now.isoformat()}, open(f"{DATA}/state.json", "w"), indent=0)
    log(f"done: {len(final)} signals, {len(new)} new buy signals, {int(time.time()-t0)}s")
    notify(new)


def notify(new):
    repo, tok = os.environ.get("REPO"), os.environ.get("GH_TOKEN")
    if not new or not repo or not tok: return
    owner = repo.split("/")[0]
    title = "🟢 فرص جديدة: " + "، ".join(r["ticker"] for r in new[:6])
    lines = [f"@{owner}", ""]
    for r in new:
        lines.append(f"**{r['ticker']}** {r['shariah']} — {r['strategy_ar']} — {r['action']} — منطقة الشراء {r['zone_low']}–{r['zone_high']}$")
    lines += ["", f"التفاصيل: https://{owner}.github.io/{repo.split('/')[1]}/"]
    try:
        subprocess.run(["gh", "issue", "create", "-R", repo, "-t", title, "-b", "\n".join(lines)], check=True, timeout=60)
    except Exception as e:
        log("notify failed:", e)


def single(t):
    t = t.strip().upper().replace(".", "-")
    fr = download([t, "EURUSD=X"], period="5y")
    out = jload(f"{DOCS}/data.json", {})
    look = [x for x in out.get("lookups", []) if x.get("ticker") != t]
    if t not in fr:
        look.insert(0, dict(ticker=t, error="ما لقيت بيانات لهذا الرمز", checked=now.isoformat()))
    else:
        d = fr[t]; f = features(d); eurusd = float(fr["EURUSD=X"]["Close"].iloc[-1]) if "EURUSD=X" in fr else None
        spy = download(["SPY"], period="5y").get("SPY")
        f["mom_pct"] = 0.5
        if spy is not None and np.isfinite(f["mom"]):
            sm = spy["Close"].to_numpy(float); f["mom_pct"] = 0.8 if f["mom"] > sm[-22] / sm[-253] - 1 else 0.4
        i = info_of(t)
        badge, note = shariah(t, i)
        m = respected_ma(d)
        hits = []
        for fn in (strat_pullback, strat_pead, strat_sale):
            try:
                s = fn(t, d, f)
                if s and (s["strategy"] != "sale" or quality_ok(i)): hits.append(enrich(t, s, {}, d, i, eurusd))
            except Exception as e:
                log("strategy error", e)
        if m:
            lo, hi = m["value"] * (1 - C["zone_below"]), m["value"] * (1 + C["zone_above"])
        base = dict(strategy="pullback", status=zone_status(f["price"], lo, hi) or "far", zone_low=lo, zone_high=hi, score=0,
                    why=(f"المتوسط المحترم: {m['ma']} يوم ({m['bounces']}/{m['touches']} ارتداد، ثقة {m['conf']})." if m else "لا يوجد متوسط يحترمه السهم بوضوح."),
                    ma_line=m["line"] if m else None) if m else None
        row = enrich(t, base, {}, d, i, eurusd) if base else dict(ticker=t, name=i.get("shortName", t), price=r2(f["price"]))
        if hits:
            row.update(action=hits[0]["action"], status=hits[0]["status"], zone_low=hits[0]["zone_low"], zone_high=hits[0]["zone_high"],
                       zone_low_eur=hits[0]["zone_low_eur"], zone_high_eur=hits[0]["zone_high_eur"], why=hits[0]["why"], strategy_ar=hits[0]["strategy_ar"])
        elif base:
            row.update(status="info", action="لا توجد إشارة شراء الآن", strategy_ar="منطقة الدعم (للمعلومة)")
            if not f["above200"]: row["why"] += " السهم تحت متوسط ٢٠٠ يوم (اتجاه هابط)."
        row.update(shariah=badge, shariah_note=note, strategies=[h["strategy_ar"] + ": " + h["action"] for h in hits],
                   verdict=("❌ غير متوافق شرعًا" if badge == "❌" else (hits[0]["action"] if hits else "لا توجد إشارة شراء الآن")),
                   checked=now.isoformat(), drawdown=r2(f["drawdown"] * 100, 1), above200=bool(f["above200"]))
        look.insert(0, row)
    out["lookups"] = look[:12]
    json.dump(out, open(f"{DOCS}/data.json", "w"), ensure_ascii=False, separators=(",", ":"))
    log(json.dumps(look[0], ensure_ascii=False)[:600])


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True); os.makedirs(DOCS, exist_ok=True)
    tk = (os.environ.get("TICKER") or (sys.argv[1] if len(sys.argv) > 1 else "")).strip()
    single(tk) if tk else full_scan()
