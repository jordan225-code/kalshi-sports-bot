"""Read-only mobile sports market research. Import real historical scores before predictions."""
import io, sqlite3, time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import streamlit as st
import bot,engine,storage

st.set_page_config(page_title='Kalshi Sports Lab v3',page_icon='📊',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>div.block-container{padding-top:1.1rem;max-width:1150px}@media(max-width:700px){div.block-container{padding:1rem .6rem}h1{font-size:1.65rem!important}}</style>',unsafe_allow_html=True)
st.title('📊 Kalshi Sports Lab v3')
st.caption('Read-only • NFL · CFB · NBA · NHL · MLB • No account access or orders')

@st.cache_resource
def db():
    c=sqlite3.connect(str(Path(__file__).with_name('markets.sqlite3')),timeout=30,check_same_thread=False)
    c.execute('PRAGMA busy_timeout=30000');c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE IF NOT EXISTS forecasts_v3(ticker TEXT PRIMARY KEY,sport TEXT,home_team TEXT,away_team TEXT,side TEXT,probability REAL,forecast_at TEXT,model_version TEXT)')
    c.commit();return c

def query(sql):return pd.read_sql_query(sql,db())

def snapshots():
    try:return query('SELECT * FROM snapshots ORDER BY time DESC')
    except Exception:return pd.DataFrame()

def outcomes():
    try:return query('SELECT * FROM outcomes')
    except Exception:return pd.DataFrame()

def save_forecast(record):
    db().execute('INSERT OR REPLACE INTO forecasts_v3 VALUES (?,?,?,?,?,?,?,?)',tuple(record[k] for k in ['ticker','sport','home_team','away_team','side','probability','forecast_at','model_version']))
    db().commit()
    if storage.enabled():storage.put('forecasts', [record])

with st.sidebar:
    st.header('Controls')
    limit=st.number_input('Markets to inspect per scan',50,5000,100,50)
    cost=st.number_input('Estimated fee + slippage ($ per $1 payout)',0.,.50,.03,.005,format='%.3f')
    edge_min=st.slider('Minimum net edge',.01,.30,.05,.01)
    if st.button('Scan open markets',use_container_width=True):
        try:bot.scan(int(limit));st.success('Scan completed.');st.rerun()
        except Exception as e:st.error(f'Scan failed: {e}')
    if st.button('Fetch recent settlements',use_container_width=True):
        try:bot.settle(int(limit));st.success('Fetched settlements.');st.rerun()
        except Exception as e:st.error(f'Settlement fetch failed: {e}')
    st.caption('Public Kalshi API only. Scans run on demand; not a background alert service.')

if 'games' not in st.session_state:st.session_state.games=None
if 'validation' not in st.session_state:st.session_state.validation=None
if 'models' not in st.session_state:st.session_state.models={}
if 'history' not in st.session_state:st.session_state.history={}

# Restore historical games across Streamlit restarts if Supabase is configured.
if st.session_state.games is None and storage.enabled():
    try:
        saved=storage.get('games',10000)
        if saved:st.session_state.games=engine.prepare(pd.DataFrame(saved))
    except Exception as e:st.warning('Remote storage unavailable: '+str(e))

tabs=st.tabs(['Overview','Historical games','Model validation','Pregame forecasts','Value alerts','Market history','Storage setup'])
snap=snapshots();settled=outcomes();latest=snap.drop_duplicates('ticker') if len(snap) else snap
with tabs[0]:
    a,b,c=st.columns(3);a.metric('Tracked markets',len(latest));b.metric('Price snapshots',len(snap));c.metric('Settled contracts',len(settled))
    st.info('Markets and settlements are not automatically paired. Market snapshots alone cannot train a team-performance model.')
    st.metric('Historical scored games loaded',len(st.session_state.games) if st.session_state.games is not None else 0)
    if len(latest):st.dataframe(latest[['ticker','title','yes_ask','no_ask','time']].head(100),hide_index=True,use_container_width=True)
with tabs[1]:
    st.subheader('Import historical final scores')
    st.write('Upload real completed games for all five leagues. Dates must be actual game dates. Each row represents one game. Do not upload future results or use market prices as labels.')
    template='sport,game_date,home_team,away_team,home_score,away_score\n'
    st.download_button('Download blank CSV template',template,'games_template.csv','text/csv')
    uploaded=st.file_uploader('Historical games CSV',type='csv')
    if uploaded and st.button('Validate and load games'):
        try:
            games=engine.prepare(pd.read_csv(uploaded))
            st.session_state.games=games;st.session_state.validation=None;st.session_state.models={}
            if storage.enabled():
                data=games.copy();data['game_date']=data.game_date.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                storage.put('games',data[engine.REQUIRED].to_dict('records'))
            st.success(f'Loaded {len(games)} completed games.')
        except Exception as e:st.error(str(e))
    if st.session_state.games is not None:
        st.dataframe(st.session_state.games.groupby('sport').size().rename('games'),use_container_width=True)
        st.download_button('Export loaded games',st.session_state.games.to_csv(index=False),'historical_games.csv')
with tabs[2]:
    st.subheader('Chronological out-of-sample test')
    st.caption('Separate model per league. Previous-game scoring averages and win rates only. Last 25% of game dates held out. Model is a basic baseline, not a validated profitable strategy.')
    if st.session_state.games is None:st.warning('Import historical games first.')
    elif st.button('Train and run holdout evaluation',use_container_width=True):
        with st.spinner('Training league models...'):
            v,m,h=engine.validate(st.session_state.games)
            st.session_state.validation=v;st.session_state.models=m;st.session_state.history=h
    v=st.session_state.validation
    if v is not None:
        if v.empty:st.warning('Insufficient history: each league needs at least 60 training examples, 10 holdout examples and enough dates. No model approved.')
        else:
            v=v.copy();v['brier']=(v.predicted_home_win-v.home_win)**2
            metrics=v.groupby('sport').agg(n=('home_win','size'),brier=('brier','mean'),actual_rate=('home_win','mean')).reset_index()
            metrics['constant_baseline_brier']=metrics.actual_rate*(1-metrics.actual_rate)
            metrics['beats_baseline']=metrics.brier<metrics.constant_baseline_brier
            st.dataframe(metrics,hide_index=True,use_container_width=True)
            st.caption('Baseline uses the test-set outcome rate (descriptive only). Repeated tuning on the same holdout invalidates its independence. Use fresh forward data for stronger evidence.')
            st.download_button('Export dated holdout predictions',v.to_csv(index=False),'holdout_predictions.csv')
with tabs[3]:
    st.subheader('Pregame home-win prediction')
    st.warning('Supported: full-game home-team win markets only, excluding ties. Not calibrated for spreads, totals, first-half results, or combos. Verify Kalshi contract rules and team mapping yourself.')
    sport=st.selectbox('League',engine.SPORTS)
    home=st.text_input('Home team — exact name from historical CSV')
    away=st.text_input('Away team — exact name from historical CSV')
    ticker=st.text_input('Matching Kalshi YES contract ticker (optional)')
    if st.button('Generate pregame estimate'):
        p=engine.upcoming_probability(sport,home.strip(),away.strip(),st.session_state.models,st.session_state.history)
        if p is None:st.error('No trained model or fewer than five historical games for a team. Run model validation first.')
        else:
            st.metric('Estimated home-win probability',f'{p:.1%}')
            st.caption('Refitted on uploaded completed games. Only valid for games occurring AFTER the latest imported game date. No injuries, lineups, or odds included.')
            if ticker.strip():
                rec=dict(ticker=ticker.strip().upper(),sport=sport,home_team=home.strip(),away_team=away.strip(),side='yes',probability=p,forecast_at=datetime.now(timezone.utc).isoformat(),model_version='rolling10-logreg-v1')
                save_forecast(rec);st.success('Forecast recorded for future evaluation; not independently certified.')
with tabs[4]:
    st.subheader('Read-only mispricing watchlist')
    st.caption('Price comparison is informational; fee allowance is a user estimate, not the exchange fee schedule.')
    try:fc=query('SELECT * FROM forecasts_v3')
    except Exception:fc=pd.DataFrame()
    if fc.empty or latest.empty:st.info('Need both a saved pregame forecast and a matching live market snapshot.')
    else:
        merged=fc.merge(latest[['ticker','yes_ask','time']],on='ticker',how='inner')
        merged['quote_time']=pd.to_datetime(merged.time,utc=True,errors='coerce')
        merged['age_minutes']=(pd.Timestamp.now(tz='UTC')-merged.quote_time).dt.total_seconds()/60
        merged['net_edge']=merged.probability-merged.yes_ask-cost
        merged['status']=merged.apply(lambda r:'STALE QUOTE' if pd.isna(r.age_minutes) or r.age_minutes>30 else ('RESEARCH ONLY' if r.net_edge>=edge_min else 'NO FLAG'),axis=1)
        st.warning('Model skill is not yet independently certified. No actionable trade alerts are enabled.')
        st.dataframe(merged[['ticker','sport','probability','yes_ask','net_edge','age_minutes','status']],hide_index=True,use_container_width=True)
        st.download_button('Export watchlist',merged.to_csv(index=False),'watchlist.csv')
with tabs[5]:
    if snap.empty:st.info('Use Controls → Scan open markets.')
    else:
        ticker=st.selectbox('Ticker',sorted(snap.ticker.unique()))
        hist=snap[snap.ticker==ticker].copy();hist['time']=pd.to_datetime(hist.time,utc=True)
        st.line_chart(hist.sort_values('time').set_index('time')[['yes_bid','yes_ask','no_bid','no_ask']])
        st.download_button('Export market snapshots',snap.to_csv(index=False),'market_snapshots.csv')
with tabs[6]:
    st.subheader('Persistent storage')
    if storage.enabled():st.success('Supabase configured. Imported historical games and new forecasts are saved remotely.')
    else:st.warning('Using local temporary SQLite + browser session. Data may be lost on Streamlit restart. Configure Supabase for persistence.')
    st.code('SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"\nSUPABASE_KEY = "YOUR_SECRET_KEY"',language='toml')
    st.caption('Add secrets in Streamlit app settings, NEVER in GitHub. Run the included schema.sql in Supabase first. A privileged secret key grants database access: keep the dashboard private or add authentication before using it. Market snapshot persistence is not included in this release.')
st.divider();st.caption('Educational research • no automated orders • no guaranteed profit • API availability varies')
