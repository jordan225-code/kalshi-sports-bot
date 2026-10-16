"""Chronological team-score model. No market prices used as training labels/features."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

SPORTS=['NFL','CFB','NBA','NHL','MLB']
REQUIRED=['sport','game_date','home_team','away_team','home_score','away_score']

def prepare(raw):
    missing=set(REQUIRED)-set(raw.columns)
    if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
    d=raw.copy(); d['sport']=d.sport.astype(str).str.upper().str.strip()
    d=d[d.sport.isin(SPORTS)].copy()
    d['game_date']=pd.to_datetime(d.game_date,utc=True,errors='coerce')
    for c in ['home_score','away_score']: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=REQUIRED).sort_values(['game_date','sport','home_team','away_team']).drop_duplicates(['sport','game_date','home_team','away_team'])
    d=d[d.home_score!=d.away_score].copy() # excludes draws: winner-only model
    if d.empty: raise ValueError('No valid completed games (draws excluded).')
    return d.reset_index(drop=True)

def feature_rows(d, minimum=5):
    # History is updated AFTER each day's games, preventing same-day score leakage.
    history={}; rows=[]
    for (sport,day),games in d.groupby(['sport','game_date'],sort=True):
        pending=[]
        for _,g in games.iterrows():
            h=history.get((sport,str(g.home_team)),[]); a=history.get((sport,str(g.away_team)),[])
            if min(len(h),len(a))>=minimum:
                def avg(v,i):return float(np.mean([x[i] for x in v[-10:]]))
                rows.append({'sport':sport,'game_date':day,'home_team':g.home_team,'away_team':g.away_team,
                    'home_scored':avg(h,0),'home_allowed':avg(h,1),'away_scored':avg(a,0),'away_allowed':avg(a,1),
                    'home_win_rate':avg(h,2),'away_win_rate':avg(a,2),'home_win':int(g.home_score>g.away_score)})
            pending.extend([((sport,str(g.home_team)),(g.home_score,g.away_score,int(g.home_score>g.away_score))),
                            ((sport,str(g.away_team)),(g.away_score,g.home_score,int(g.away_score>g.home_score)))])
        for key,item in pending:history.setdefault(key,[]).append(item)
    return pd.DataFrame(rows),history

FEATURES=['home_scored','home_allowed','away_scored','away_allowed','home_win_rate','away_win_rate']

def validate(d, min_train=60, test_fraction=.25):
    feats,history=feature_rows(d)
    if feats.empty:return pd.DataFrame(),{},history
    predictions=[]; models={}
    for sport,group in feats.groupby('sport'):
        group=group.sort_values('game_date').reset_index(drop=True)
        # Test on the most recent 25% of distinct dates; train only on earlier dates.
        days=sorted(group.game_date.unique())
        if len(days)<8:continue
        cut=days[max(1,int(len(days)*(1-test_fraction)))]
        train=group[group.game_date<cut]; test=group[group.game_date>=cut]
        if len(train)<min_train or len(test)<10 or train.home_win.nunique()<2:continue
        model=LogisticRegression(max_iter=1000)
        model.fit(train[FEATURES],train.home_win)
        p=model.predict_proba(test[FEATURES])[:,list(model.classes_).index(1)]
        result=test[['sport','game_date','home_team','away_team','home_win']].copy()
        result['predicted_home_win']=p;result['split']='chronological_holdout';predictions.append(result)
        # Production refit after holdout; never evaluate this refit on the holdout.
        if group.home_win.nunique()==2:
            production=LogisticRegression(max_iter=1000).fit(group[FEATURES],group.home_win)
            models[sport]=production
    if not predictions:return pd.DataFrame(),models,history
    return pd.concat(predictions,ignore_index=True),models,history

def upcoming_probability(sport,home,away,models,history):
    sport=sport.upper(); model=models.get(sport)
    h=history.get((sport,home),[]);a=history.get((sport,away),[])
    if model is None or min(len(h),len(a))<5:return None
    def avg(v,i):return float(np.mean([x[i] for x in v[-10:]]))
    features=pd.DataFrame([dict(zip(FEATURES,[avg(h,0),avg(h,1),avg(a,0),avg(a,1),avg(h,2),avg(a,2)]))])
    return float(model.predict_proba(features)[0,list(model.classes_).index(1)])
