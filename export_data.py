from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

BASE=Path(r'G:\shucai\Huawei_dual_official_all_20260827')
SITE=Path(r'G:\shucai\human-trajectory-preview')
EX=Path(r'G:\shucai\human_trajectory_turn_multihypothesis_20260917')
def arr(x): return np.round(np.asarray(x,float),4).tolist()
def dump(path,data): path.write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False),encoding='utf8')
def main():
    (SITE/'data').mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(BASE/'spatial_alignment_v5/spatial_alignment_manifest.csv')
    windows=pd.read_parquet(BASE/'direction_cosine_v1/direction_windows_2s.parquet')
    v14=pd.read_csv(BASE/'end_to_end_trajectory_v1/v14_mapframe_continuous_direction/oof_predictions.csv.gz')
    fresh=pd.read_csv(EX/'all_oof_predictions.csv.gz')
    fresh=fresh[(fresh.seed==20260903)&fresh.scheme.isin(['SELECTED_09','SELECTED_10'])]
    cards=[]
    for _,m in manifest.iterrows():
        uid=int(m.uid); source=BASE/'spatial_alignment_v5/sequences_90hz'/f'uid_{uid}_dual_spatial_v5.csv.gz'
        if not source.exists(): continue
        df=pd.read_csv(source); t=df.time_s.to_numpy(); clock=np.arange(t[0],t[-1]+.00001,.1)
        def sample(cols): return np.column_stack([np.interp(clock,t,df[c]) for c in cols])
        pos=sample(['gt_x_m','gt_y_m','gt_z_m']); pos-=pos[0]
        vel=sample(['gt_vx_mps','gt_vy_mps','gt_vz_mps'])
        series={'gt':{'linear':arr(np.gradient(vel,.1,axis=0))}}
        for dev,frame in [('phone','gt'),('watch','gravity')]:
            series[dev]={kind:arr(sample([f'{dev}_{stem}_{frame}_{a}' for a in 'xyz'])) for kind,stem in [('raw','acc'),('linear','linear_acc'),('gyro','gyro')]}
        lag={};g=np.linalg.norm(np.asarray(series['gt']['linear'])[:,[0,2]],axis=1)
        for dev in ['phone','watch']:
            a=np.linalg.norm(np.asarray(series[dev]['linear'])[:,[0,2]],axis=1);scores=[]
            for k in range(-10,11):
                left=g[-k:] if k<0 else g[:-k] if k>0 else g
                right=a[:k] if k<0 else a[k:] if k>0 else a
                value=float(np.corrcoef(left,right)[0,1]) if np.std(left)>1e-9 and np.std(right)>1e-9 else 0.
                scores.append((k,value))
            k,c=max(scores,key=lambda x:x[1]);lag[dev]={'candidate_lag_s':k/10,'best_correlation':c,'zero_lag_correlation':scores[10][1],'at_search_boundary':abs(k)==10,'applied_lag_s':0,'sign':'positive means device signal occurs later than GT','method':'10Hz horizontal linear-acceleration magnitude; search ±1s; QA only'}
        metrics={};tracks={}; w=windows[windows.uid==uid][['window','start_s','end_s','duration_s']]
        for name,p in [('v14_phone',v14[v14['mode']=='phone_only']),('09_phone',fresh[(fresh.scheme=='SELECTED_09')&(fresh['mode']=='phone')]),('09_dual',fresh[(fresh.scheme=='SELECTED_09')&(fresh['mode']=='dual')]),('10_phone',fresh[(fresh.scheme=='SELECTED_10')&(fresh['mode']=='phone')]),('10_dual',fresh[(fresh.scheme=='SELECTED_10')&(fresh['mode']=='dual')])]:
            q=p[p.uid==uid].merge(w,on='window',validate='one_to_one').sort_values('window')
            if len(q)==0: continue
            # Use matched 2-second labels for evaluation. Reset at a missing window.
            pts=[]; pp=np.zeros(2);gg=np.zeros(2); errors=[]; cosine=[];ratios=[];blocks=[]; last=None;speed=[]
            for r in q.itertuples():
                if last is None or r.window!=last+1:
                    pp=np.zeros(2);gg=np.zeros(2);blocks.append([])
                    pts.append([float(r.start_s),0.,0.,True])
                pv=np.array([r.pred_vx_mps,r.pred_vz_mps]); gv=np.array([r.gt_vx_mps,r.gt_vz_mps])
                # Existing benchmark integrates nominal 2s windows, not optical dense steps.
                pp+=pv*2; gg+=np.array([r.gt_dx_m,r.gt_dz_m]);e=float(np.linalg.norm(pp-gg));errors.append(e);blocks[-1].append((e,np.linalg.norm(pv)*2,float(r.gt_distance_m)))
                pts.append([float(r.end_s),*pp.tolist(),False]);speed.append([float(r.end_s),float(np.linalg.norm(pv))])
                if np.linalg.norm(gv)>.12: cosine.append(float(np.dot(pv,gv)/max(np.linalg.norm(pv)*np.linalg.norm(gv),1e-9)))
                last=r.window
            for b in blocks: ratios.append(sum(r[1] for r in b)/max(sum(r[2] for r in b),1e-9))
            tracks[name]={'points':pts,'speed':speed,'fold':int(q.fold.iloc[0]),'windows':len(q),'selected_sources':sorted(q.selected_source.dropna().unique().tolist()) if 'selected_source' in q else [],'information':'GT坐标辅助历史基线' if name=='v14_phone' else 'q_nav输入；嵌套验证选路；seed 20260903'}
            metrics[name]={'cosine':float(np.mean(cosine)) if cosine else None,'opposite':float(np.mean(np.array(cosine)<0)) if cosine else None,'ade':float(np.mean(errors)),'fde':float(np.mean([b[-1][0] for b in blocks])),'path_ratio':float(np.median(ratios)),'blocks':len(blocks),'active_windows':len(cosine)}
        segs=[]
        # GT-selected straight segments are diagnostics only, never model input.
        for st in np.arange(clock[0]+2,clock[-1]-6,6):
            ii=np.flatnonzero((clock>=st)&(clock<=st+6));xz=pos[ii][:,[0,2]];distance=np.linalg.norm(xz[-1]-xz[0]);length=np.linalg.norm(np.diff(xz,axis=0),axis=1).sum()
            if distance>1.5 and distance/max(length,1e-9)>.92: segs.append({'start':int(ii[0]),'end':int(ii[-1]),'start_s':round(float(st),2),'end_s':round(float(st+6),2),'length_m':round(float(distance),2),'bearing_deg':round(float(np.degrees(np.arctan2(*(xz[-1]-xz[0])[::-1]))),1)})
        def clean(v):
            if isinstance(v,(np.bool_,bool)):return bool(v)
            if pd.isna(v):return None
            return v.item() if hasattr(v,'item') else v
        meta={k:clean(m[k]) for k in ['source_sync_method','phone_sync_status','watch_sync_status','phone_sync_corr','watch_sync_corr','phone_directional_node_valid','watch_directional_node_valid','watch_world_yaw_verified','phone_calibration_status','phone_q_coverage','watch_q_coverage','huawei_to_gt_yaw_deg']}
        data={'uid':uid,'label':str(m.ly_file),'time_s':arr(clock),'position':arr(pos),'series':series,'tracks':tracks,'metrics':metrics,'segments':segs[:80],'metadata':meta,'sample_hz':10,'source_hz':90,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'missing':['逐marker位置','每帧IMU原始有效性掩码'],'coordinate_contract':{'gt':'GT世界XYZ；+Y向上；首GT质心为原点','phone':'v5 GT坐标；方向可信度见元数据','watch':'v5去倾斜局部坐标；绝对水平航向未经核验，不能当GT世界方向'},'gt_acc_method':'GT速度数值微分；10Hz展示采样，非原始传感器测量'}
        data['lag_diagnostics']=lag
        dump(SITE/'data'/f'{uid}.json',data)
        cards.append({'uid':uid,'label':str(m.ly_file),'date':str(m.date),'duration_s':round(float(clock[-1]-clock[0]),1),'frames':len(clock),'file':f'data/{uid}.json','watch_world_yaw_verified':bool(meta['watch_world_yaw_verified'])})
        print(uid,len(clock),list(tracks),flush=True)
    dump(SITE/'data/index.json',{'schema':'human_trajectory_preview_v1','sequences':cards,'default_uid':140006,'paper_standard':json.loads((EX/'acceptance_standard_v1.json').read_text(encoding='utf8'))})
    print('Complete',len(cards),flush=True)
if __name__=='__main__':main()
