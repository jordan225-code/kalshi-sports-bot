"""Optional Supabase REST persistence. No write access to Kalshi."""
import os,requests

def config():
    try:
        import streamlit as st
        url=st.secrets.get('SUPABASE_URL','');key=st.secrets.get('SUPABASE_KEY','')
    except Exception:url=key=''
    return url or os.getenv('SUPABASE_URL',''),key or os.getenv('SUPABASE_KEY','')

def enabled():return all(config())

def headers():
    _,key=config();return {'apikey':key,'Authorization':'Bearer '+key,'Content-Type':'application/json','Prefer':'resolution=merge-duplicates,return=minimal'}

def put(table,records):
    if not enabled():raise RuntimeError('Configure Supabase secrets first.')
    if not records:return
    url,_=config()
    for i in range(0,len(records),200):
        r=requests.post(url.rstrip('/')+'/rest/v1/'+table,json=records[i:i+200],headers=headers(),timeout=25)
        r.raise_for_status()

def get(table,limit=10000):
    if not enabled():return []
    url,_=config();r=requests.get(url.rstrip('/')+'/rest/v1/'+table,params={'select':'*','limit':limit},headers=headers(),timeout=25);r.raise_for_status();return r.json()
