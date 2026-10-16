"""Read-only mobile-friendly research dashboard. Run: streamlit run app.py"""
import io, sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
import bot

st.set_page_config(page_title='Kalshi Sports Lab',page_icon='📊',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>div.block-container{padding-top:1.4rem;max-width:1100px} [data-testid="stMetric"]{border:1px solid #3b4351;border-radius:12px;padding:12px} @media(max-width:700px){h1{font-size:1.8rem!important}div.block-container{padding:1rem .7rem}}</style>''',unsafe_allow_html=True)
st.title('📊 Kalshi Sports Lab')
st.caption('Read-only sports market research • Not a trade recommendation • No orders or account access')

def conn():return bot.db()
def read(sql,params=()):
    with conn() as c:return pd.read_sql_query(sql,c,params=params)
with conn() as c:
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts(ticker TEXT,side TEXT,probability REAL,forecast_at TEXT,source TEXT,PRIMARY KEY(ticker,side,forecast_at))''')
    c.commit()

with st.sidebar:
    st.header('Controls')
    limit=st.number_input('Markets to inspect per scan',200,10000,2000,200)
    fee=st.number_input('Fee + slippage allowance per $1 payout ($)',0.0,0.5,0.03,0.005,format='%.3f')
    min_edge=st.slider('Minimum estimated net edge',0.01,0.30,0.05,0.01)
    st.caption('Fee/slippage is a conservative user-set approximation, not an exchange fee calculation.')
    if st.button('Scan open markets',use_container_width=True):
        try:bot.scan(limit);st.success('Snapshot saved. Refresh dashboard to see updates.')
        except Exception as e:st.error(f'Could not scan public API: {e}')
    if st.button('Fetch recent settlements',use_container_width=True):
        try:bot.settle(limit);st.success('Recent settled outcomes updated.')
        except Exception as e:st.error(f'Could not fetch settlements: {e}')

tabs=st.tabs(['Overview','Market history','Model validation','Value alerts','Import forecasts'])
snap=read('SELECT * FROM snapshots ORDER BY time DESC')
out=read('SELECT * FROM outcomes')
fc=read('SELECT * FROM forecasts')
latest=snap.drop_duplicates('ticker') if len(snap) else snap
with tabs[0]:
    a,b,c=st.columns(3)
    a.metric('Tracked markets',len(latest));b.metric('Saved snapshots',len(snap));c.metric('Resolved markets',len(out))
    st.info('Historical collection starts when you run scans. It is not a complete historical archive. Sports discovery is keyword-based and can miss markets.')
    if len(latest):
        q=st.text_input('Find a sport, team, or ticker')
        view=latest[latest['title'].str.contains(q,case=False,na=False)|latest['ticker'].str.contains(q,case=False,na=False)] if q else latest
        st.dataframe(view[['ticker','title','yes_ask','no_ask','volume','time']],hide_index=True,use_container_width=True)
    else:st.write('No snapshots yet. Open Controls → Scan open markets to start collecting.')
with tabs[1]:
    if len(snap):
        ticker=st.selectbox('Select market',sorted(snap.ticker.unique()))
        history=snap[snap.ticker==ticker].copy().sort_values('time')
        history['time']=pd.to_datetime(history.time,utc=True)
        st.line_chart(history.set_index('time')[['yes_bid','yes_ask','no_bid','no_ask']])
        st.dataframe(history,hide_index=True,use_container_width=True)
        st.download_button('Export all snapshots CSV',snap.to_csv(index=False),'snapshots.csv','text/csv')
    else:st.info('Scan markets multiple times to build a price history.')

# Evaluation uses ONLY forecast timestamps preceding recorded outcome collection timestamps.
joined=pd.DataFrame()
if len(fc) and len(out):
    joined=fc.merge(out[['ticker','result','settled_at']],on='ticker',how='inner')
    joined['forecast_at']=pd.to_datetime(joined.forecast_at,utc=True,errors='coerce')
    joined['settled_at']=pd.to_datetime(joined.settled_at,utc=True,errors='coerce')
    joined=joined[(joined.forecast_at<joined.settled_at)&joined.probability.between(0,1)].copy()
    joined['actual']=(joined.side.str.lower()==joined.result.str.lower()).astype(int)
    joined['brier']=(joined.probability-joined.actual)**2
    # One prediction per ticker and side to avoid repeated scans inflating evidence.
    joined=joined.sort_values('forecast_at').drop_duplicates(['ticker','side'],keep='first')
with tabs[2]:
    st.subheader('Out-of-sample validation monitor')
    st.caption('This is a performance monitor, NOT proof of independent modeling or genuine out-of-sample validation. Ensure forecast inputs exclude post-event information and are produced by a separately validated sports model.')
    if len(joined):
        n=len(joined); brier=joined.brier.mean(); baseline=((joined.actual-joined.actual.mean())**2).mean()
        x,y,z=st.columns(3);x.metric('Evaluated forecasts',n);y.metric('Brier score',f'{brier:.3f}');z.metric('Constant-rate baseline',f'{baseline:.3f}')
        st.write('Lower Brier scores are better. The baseline uses outcomes from this evaluation set and is descriptive, not a prospective benchmark.')
        joined['bucket']=(joined.probability*10).clip(upper=9.999).astype(int)/10
        cal=joined.groupby('bucket').agg(predicted=('probability','mean'),observed=('actual','mean'),count=('actual','size')).reset_index()
        st.dataframe(cal,hide_index=True,use_container_width=True)
        st.dataframe(joined[['ticker','side','probability','actual','brier','forecast_at']],hide_index=True,use_container_width=True)
    else:st.warning('No eligible forecast/outcome pairs yet. Do not treat signals as validated.')
with tabs[3]:
    st.subheader('Potential mispricing — gated alerts')
    st.caption('Requires ≥100 eligible resolved predictions, better Brier score than descriptive baseline, and current quotes less than 30 minutes old. These are screening gates, not a guarantee of calibration.')
    valid=len(joined)>=100 and joined.brier.mean()<((joined.actual-joined.actual.mean())**2).mean()
    if not valid:st.warning('Alerts locked: insufficient evidence of forecasting skill. Import independent forecasts and accumulate resolved predictions.')
    if len(latest) and len(fc):
        forecasts=fc.sort_values('forecast_at').drop_duplicates(['ticker','side'],keep='last')
        records=[]
        for _,r in forecasts.iterrows():
            match=latest[latest.ticker==r.ticker]
            if match.empty:continue
            m=match.iloc[0];ask=m.yes_ask if r.side=='yes' else m.no_ask
            if pd.isna(ask):continue
            age=(datetime.now(timezone.utc)-pd.to_datetime(m.time,utc=True).to_pydatetime()).total_seconds()/60
            edge=float(r.probability)-float(ask)-fee
            records.append(dict(ticker=r.ticker,side=r.side,model_probability=r.probability,ask=ask,net_edge=edge,quote_age_minutes=round(age,1),status=('REVIEW' if valid and age<=30 and edge>=min_edge else 'NO ALERT')))
        if records:
            result=pd.DataFrame(records).sort_values('net_edge',ascending=False)
            st.dataframe(result,hide_index=True,use_container_width=True)
            st.download_button('Export research alerts',result.to_csv(index=False),'research_alerts.csv','text/csv')
        else:st.info('No forecast tickers match current stored quotes.')
    else:st.info('Market snapshots and independently generated forecasts are needed.')
with tabs[4]:
    st.subheader('Import independent probability forecasts')
    st.write('CSV fields: `ticker,side,model_probability,forecast_at,source`. Times must be UTC ISO-8601. Only genuine predictions made at that time should be uploaded; do not backdate.')
    st.code('ticker,side,model_probability,forecast_at,source\nEXAMPLE-TICKER,yes,0.61,2026-10-16T12:00:00+00:00,my-independent-model',language='csv')
    upload=st.file_uploader('Upload forecasts CSV',type='csv')
    if upload and st.button('Validate and save forecasts'):
        try:
            f=pd.read_csv(upload)
            required={'ticker','side','model_probability','forecast_at','source'}
            if not required.issubset(f.columns):raise ValueError(f'Missing columns: {required-set(f.columns)}')
            f['side']=f.side.str.lower().str.strip()
            if not f.side.isin(['yes','no']).all():raise ValueError('Side must be yes or no')
            if not pd.to_numeric(f.model_probability,errors='coerce').between(0,1).all():raise ValueError('Probability must be between 0 and 1')
            dates=pd.to_datetime(f.forecast_at,utc=True,errors='coerce')
            if dates.isna().any() or (dates>pd.Timestamp.now(tz='UTC')).any():raise ValueError('Dates must be valid, not in the future')
            if f[['ticker','source']].isna().any().any():raise ValueError('Ticker/source required')
            rows=[(str(r.ticker),r.side,float(r.model_probability),d.isoformat(),str(r.source)) for (_,r),d in zip(f.iterrows(),dates)]
            with conn() as c:c.executemany('INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?)',rows);c.commit()
            st.success(f'Saved {len(rows)} forecasts. Reload page to recalculate scores.')
        except Exception as e:st.error(str(e))
st.divider()
st.caption('Educational research only. No order submission, authentication, auto-betting, or promises of profit. API availability and sports coverage may vary.')
