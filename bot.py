"""Read-only Kalshi sports market recorder and price-value screener."""
import argparse, csv, json, sqlite3, time, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE='https://external-api.kalshi.com/trade-api/v2'
DB=Path(__file__).with_name('markets.sqlite3')
SPORTS=('NFL','NBA','MLB','NHL','NCAA','FOOTBALL','BASEBALL','BASKETBALL','HOCKEY','SOCCER','TENNIS','CFB')

def get(path, params=None):
    url=BASE+path+('?' + urllib.parse.urlencode(params) if params else '')
    req=urllib.request.Request(url,headers={'User-Agent':'SportsResearchBot/0.1'})
    with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)

def db():
    c=sqlite3.connect(DB)
    c.execute('CREATE TABLE IF NOT EXISTS snapshots (time TEXT,ticker TEXT,title TEXT,yes_bid REAL,yes_ask REAL,no_bid REAL,no_ask REAL,volume REAL,PRIMARY KEY(time,ticker))')
    c.execute('CREATE TABLE IF NOT EXISTS outcomes (ticker TEXT PRIMARY KEY,title TEXT,result TEXT,settled_at TEXT)')
    return c

def num(m,key):
    v=m.get(key+'_dollars')
    if v is None and m.get(key) is not None:v=float(m[key])/100
    try:return float(v) if v is not None else None
    except (ValueError,TypeError):return None

def sports(m):
    return any(s in (' '.join(str(m.get(k,'')) for k in ('ticker','event_ticker','title','subtitle','category'))).upper() for s in SPORTS)

def scan(limit):
    conn=db(); stamp=datetime.now(timezone.utc).isoformat(); cursor=None; seen=0; stored=0
    while seen<limit:
        params={'status':'open','limit':min(200,limit-seen)}
        if cursor:params['cursor']=cursor
        response=get('/markets',params); markets=response.get('markets',[])
        if not markets:break
        for m in markets:
            if not sports(m):continue
            vals=[num(m,k) for k in ('yes_bid','yes_ask','no_bid','no_ask')]
            conn.execute('INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?,?,?,?,?)',(stamp,m['ticker'],m.get('title',''),*vals,float(m.get('volume_fp') or m.get('volume') or 0)))
            stored+=1
        seen+=len(markets); cursor=response.get('cursor')
        if not cursor:break
        time.sleep(.2)
    conn.commit();conn.close()
    print(f'Checked {seen} open markets; stored {stored} sport-matching snapshots in {DB}.')
    print('Discovery is keyword-based and may miss sports markets. No orders placed.')

def settle(limit):
    conn=db();count=0;cursor=None;seen=0
    while seen<limit:
        params={'status':'settled','limit':min(200,limit-seen)}
        if cursor:params['cursor']=cursor
        response=get('/markets',params);ms=response.get('markets',[])
        if not ms:break
        for m in ms:
            if sports(m) and str(m.get('result','')).lower() in ('yes','no'):
                conn.execute('INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)',(m['ticker'],m.get('title',''),m['result'],datetime.now(timezone.utc).isoformat()));count+=1
        seen+=len(ms);cursor=response.get('cursor')
        if not cursor:break
        time.sleep(.2)
    conn.commit();conn.close();print(f'Saved {count} settled results; recent API pages only, not a full historical backfill.')

def screen(csv_path,fee):
    # CSV model probabilities MUST come from an independent, out-of-sample validated model.
    conn=db();rows=[]
    with open(csv_path,newline='') as f:
        for r in csv.DictReader(f):
            ticker=r['ticker'].strip(); side=r['side'].strip().lower();p=float(r['model_probability'])
            if side not in ('yes','no') or not 0<=p<=1:continue
            latest=conn.execute('SELECT title,yes_ask,no_ask FROM snapshots WHERE ticker=? ORDER BY time DESC LIMIT 1',(ticker,)).fetchone()
            if not latest:continue
            title,yesask,noask=latest; price=yesask if side=='yes' else noask
            if price is None or not 0<price<1:continue
            edge=p-price-fee
            rows.append((edge,ticker,side,p,price,title))
    rows.sort(reverse=True)
    if not rows:print('No matching price snapshots. Run scan first.');return
    print('Research signals only. Fee input is a conservative per-contract placeholder, not Kalshi exact fee calculation.')
    for edge,ticker,side,p,price,title in rows:
        label='REVIEW' if edge>=.05 else 'SKIP'
        print(f'{label:6} {ticker:40.40} {side.upper():3} model={p:.1%} ask={price:.1%} net_edge={edge:+.1%} {title[:45]}')
    conn.close()

if __name__=='__main__':
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('scan');s.add_argument('--limit',type=int,default=1000)
    s=sub.add_parser('settle');s.add_argument('--limit',type=int,default=1000)
    s=sub.add_parser('screen');s.add_argument('probabilities');s.add_argument('--fee',type=float,default=.02,help='assumed dollars per contract')
    args=a.parse_args()
    if args.cmd=='scan':scan(args.limit)
    elif args.cmd=='settle':settle(args.limit)
    else:screen(args.probabilities,args.fee)
