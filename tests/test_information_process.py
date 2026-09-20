import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from smoke_moe import synthetic_archive
from smoke_information_process import synthetic_information
from climate_diffusion.physical_information import terrain_slope,load_information,fit_information
from climate_diffusion.information_process import InformationProcess,fair_crps,gradient_diagnostics
from climate_diffusion.train_information_process import data_contract,Windows
from climate_diffusion.manifold_moe import ManifoldMoEConfig
from climate_diffusion.moe_data import load_moe_archive,field_grid
from climate_diffusion.physical_information import prepare,digest,_field
from climate_diffusion.train_information_process import load_checkpoint,write_json
from climate_diffusion.information_process import FORMAT
from dataclasses import asdict
import json
import xarray as xr

@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(7)
    archive,_=synthetic_archive(tmp_path,count=320);info=synthetic_information(archive,tmp_path)
    states,_,schema=load_moe_archive(archive)
    cfg=ManifoldMoEConfig(state_dim=states.shape[1],grid=field_grid(schema),horizon_steps=20,step_hours=6,
        history_steps=6,history_stride=1,manifold_dim=4,hidden_dim=24,context_dim=8,num_experts=2,
        expert_latent_dim=8,gate_hidden_dim=12,forecast_dynamics='recurrent_residual')
    d=data_contract(archive,info,'enriched',cfg)
    model=InformationProcess(cfg,schema,d['mean'],d['scale'],d['statistics'],d['information_metadata'])
    x=torch.tensor((states[:d['train_end']]-d['mean'])/d['scale'])
    model.core.physics.fit(x)
    ds=Windows(states,d['times'],cfg,[0,1],d['mean'],d['scale'],schema,information=d['information'])
    batch={k:torch.stack([ds[0][k],ds[1][k]]) for k in ds[0]}
    return model,batch,d,archive,info

def test_sidecar_stats_split_contract(prepared):
    m,b,d,archive,info=prepared
    assert b['information'].shape==(2,224) and b['targets'].shape==(2,20,128)
    assert torch.all(b['dt_hours']==6)
    raw,meta=load_information(info,archive,d['times'],d['schema'])
    a=fit_information(raw,meta,d['train_end'],d['schema'])
    raw[d['train_end']:]+=999
    z=fit_information(raw,meta,d['train_end'],d['schema'])
    assert all(np.array_equal(x,y) for x,y in zip(a,z))
    with pytest.raises(ValueError,match='requires'):data_contract(archive,None,'enriched',m.config)

def test_spherical_slope():
    lat=np.array([-60.,-20.,20.,60.]);lon=np.arange(8)*45.
    height=np.broadcast_to(6371000*np.deg2rad(lat)[:,None],(4,8))
    assert np.allclose(terrain_slope(height,lat,lon),1)
    with pytest.raises(ValueError):terrain_slope(height,np.array([-90,-20,20,90]),lon)

def test_full20_sampler_identity_gradient_no_future(prepared):
    m,b,d,_,_=prepared;noise=torch.randn(2,4,4);trace=[]
    output=m.rollout(b['history'],b['information'],noise=noise,auxiliary=True,trace=trace)
    assert output.shape==(2,4,21,128)
    assert torch.equal(trace[1]['input'],trace[0]['output'])
    assert not torch.equal(output[:,0,1:],output[:,1,1:])
    prefix=m.rollout(b['history'],b['information'],noise=noise,auxiliary=True,steps=2)
    assert torch.allclose(prefix,output[:,:,:3])
    truth=torch.cat((b['origin'][:,None],b['targets']),1)
    scores=m.scores(output,truth,b['dt_hours'],b['pair_observed_mask'])
    for key in ['state_crps','transition_crps','loss_trajectory']:
        grads=torch.autograd.grad(scores[key],list(m.a_sampler.parameters())+list(m.information.parameters()),retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    scores['loss_trajectory'].backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.manifold.decoder.parameters())
    assert all(p.grad is None for p in m.core.experts.parameters())
    b['information_targets'].fill_(999)
    assert torch.allclose(prefix,m.rollout(b['history'],b['information'],noise=noise,auxiliary=True,steps=2))

def test_seal_forecast_invariance_and_frozen_B(prepared):
    m,b,d,_,_=prepared;noise=torch.randn(2,4,4)
    before=m.rollout(b['history'],b['information'],noise=noise,auxiliary=True,steps=2)
    x=torch.tensor((d['states'][:d['train_end']]-d['mean'])/d['scale'])
    c=torch.tensor(d['information'][:d['train_end']]);m.seal(x,c)
    after=m.rollout(b['history'],b['information'],noise=noise,auxiliary=True,steps=2)
    assert torch.allclose(before,after,atol=3e-5,rtol=1e-4)
    with pytest.raises(ValueError):m.seal(x,c)
    m.set_phase('B');saved={k:v.clone() for k,v in m.core.manifold.state_dict().items()}
    output=m.rollout(b['history'],b['information'],noise=noise,steps=2)
    output.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.experts.parameters())
    assert all(p.grad is None for p in m.information.parameters())
    torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01).step()
    assert all(torch.equal(v,m.core.manifold.state_dict()[k]) for k,v in saved.items())

def test_fair_crps_and_nontrivial_information(prepared):
    x=torch.tensor([[[-1.],[1.]]],requires_grad=True);y=torch.zeros(1,1)
    assert fair_crps(x,y)==0
    with pytest.raises(ValueError):fair_crps(x[:,:1],y)
    m,b,_,_,_=prepared;losses=m.geometry_losses(b,b['information'])
    assert losses['static_l2']>0 and losses['information_geometry']>0
    grads=gradient_diagnostics({k:losses[k] for k in ['reconstruction','static_l2']},
                              {'encoder':m.core.manifold.encoder.parameters()})
    assert grads['gradient/reconstruction/encoder']>0 and grads['gradient/static_l2/encoder']>0

def test_units_pressure_geopotential_and_mask(prepared,tmp_path):
    m,b,d,archive,info=prepared
    with xr.open_dataset(tmp_path/'synthetic-information.nc') as src:ds=src.load()
    ds['z500']=ds['z500']*9.80665;ds['z500'].attrs['units']='m2 s-2'
    raw=tmp_path/'geopotential.nc';ds.to_netcdf(raw,engine='scipy')
    converted=prepare(archive,raw,tmp_path/'converted.npz')
    a,meta=load_information(info,archive,d['times'],d['schema'])
    z,_=load_information(converted,archive,d['times'],d['schema'])
    assert np.allclose(a,z,atol=.001)
    field=xr.Dataset({'z':(('level','lat','lon'),np.ones((2,4,8)))},coords={'level':[50000,85000]})
    field.level.attrs['units']='Pa';assert _field(field,'z500').shape==(4,8)
    field.level.attrs['units']='unknown'
    with pytest.raises(ValueError):_field(field,'z500')
    with np.load(info) as f:payload={k:f[k] for k in f.files}
    payload['observed_mask'][0,0]=0
    bad=tmp_path/'bad-mask.npz';np.savez_compressed(bad,**payload)
    with pytest.raises(ValueError,match='Missing'):load_information(bad,archive,d['times'],d['schema'])
    payload['observed_mask'][0,0]=1;meta['variables'][0]['unit']='Pa'
    payload['metadata_json']=json.dumps(meta);np.savez_compressed(bad,**payload)
    with pytest.raises(ValueError,match='units'):load_information(bad,archive,d['times'],d['schema'])

def test_checkpoint_roundtrip_and_legacy_rejection(prepared,tmp_path):
    m,b,d,_,_=prepared
    payload={'format':FORMAT,'config':asdict(m.config),'schema':d['schema'],'mean':d['mean'],'scale':d['scale'],
             'statistics':d['statistics'],'information_metadata':d['information_metadata'],'model':m.state_dict(),'stage':'A'}
    path=tmp_path/'new.pt';torch.save(payload,path)
    write_json(path.with_suffix('.manifest.json'),{'checkpoint_sha256':digest(path)})
    other,_=load_checkpoint(path);noise=torch.randn(2,4,4)
    assert torch.equal(m.rollout(b['history'],b['information'],auxiliary=True,steps=2,noise=noise),
                       other.rollout(b['history'],b['information'],auxiliary=True,steps=2,noise=noise))
    payload['format']='legacy';torch.save(payload,path)
    with pytest.raises(ValueError,match='legacy'):load_checkpoint(path)

def test_tendency_units_and_pairmask(prepared):
    m,b,_,_,_=prepared;truth=torch.cat((b['origin'][:,None],b['targets']),1)
    pred=truth[:,None].repeat(1,4,1,1)
    score=m.scores(pred,truth,b['dt_hours'],b['pair_observed_mask'])
    assert score['transition_crps']==0 and score['loss_delta']==0
    pred=pred+torch.randn_like(pred)*.01
    score6=m.scores(pred,truth,b['dt_hours'],b['pair_observed_mask'])
    score12=m.scores(pred,truth,2*b['dt_hours'],b['pair_observed_mask'])
    assert torch.allclose(score6['transition_crps'],2*score12['transition_crps'])
    mask=b['pair_observed_mask'].clone();mask[0,0,0]=False
    with pytest.raises(ValueError):m.scores(pred,truth,b['dt_hours'],mask)

def test_extra_information_really_conditions_A(prepared):
    m,b,_,_,_=prepared;noise=torch.randn(2,4,4)
    c=b['information'];changed=c.clone();changed[:,:32]+=2
    assert not torch.allclose(m.raw_encode(b['origin'],c),m.raw_encode(b['origin'],changed))
    a=m.rollout(b['history'],c,auxiliary=True,noise=noise,steps=2)
    z=m.rollout(b['history'],changed,auxiliary=True,noise=noise,steps=2)
    assert torch.equal(a[:,:,0],z[:,:,0])  # observed origin offset is preserved
    assert not torch.allclose(a[:,:,1:],z[:,:,1:])


def test_pooled_rms_is_not_mean_case_rms():
    from climate_diffusion.information_forecast import aggregate_scores
    rows=[dict(rmse=error,mean_state=error**2,spread=spread,ensemble_variance=spread**2,
               persistence_rmse=error,persistence_mse=error**2,state_crps=error)
          for error,spread in ((1.,2.),(3.,4.))]
    scores=aggregate_scores(rows)
    assert scores['rmse']==pytest.approx(np.sqrt(5))
    assert scores['persistence_rmse']==pytest.approx(np.sqrt(5))
    assert scores['spread']==pytest.approx(np.sqrt(10))
    assert scores['mean_case_rmse']==2 and scores['mean_case_spread']==3
    assert scores['state_crps']==2
    with pytest.raises(ValueError,match='No held-out'):aggregate_scores([])


def test_report_cannot_overwrite_forecast(tmp_path):
    from climate_diffusion.information_forecast import evaluate
    output=tmp_path/'same.npz'
    with pytest.raises(ValueError,match='distinct'):
        evaluate('not-read.pt','not-read.npz',output,forecast_output=output)
    assert not output.exists()


@pytest.mark.parametrize('stage',['A','B','C'])
def test_logged_weights_reproduce_loss_and_stage_gradients(prepared,stage):
    from types import SimpleNamespace
    from climate_diffusion.train_information_process import batch_loss
    m,b,d,_,_=prepared
    if stage!='A':
        m.seal(torch.tensor((d['states'][:d['train_end']]-d['mean'])/d['scale']),
               torch.tensor(d['information'][:d['train_end']]))
    m.set_phase(stage)
    before={k:v.clone() for k,v in m.state_dict().items()}
    args=SimpleNamespace(members=2,tau_steps=1,curriculum_interval=1,profile='process',
                         loss_weights=None,info_scale=d['information_scale'],
                         info_tendency_scale=d['information_tendency_scale'],b_member_weight=.001)
    streams={k:torch.Generator().manual_seed(seed) for k,seed in [('fm',11),('ensemble',29)]}
    values=batch_loss(m,b,args,6,streams)
    summed=sum(v for k,v in values.items() if k.startswith('weighted_'))
    assert torch.allclose(summed,values['loss'],rtol=1e-6,atol=1e-6)
    if stage=='A':
        for k in ('state_crps','transition_crps','loss_trajectory','static_l2','info_distribution'):
            assert values['weighted_'+k].abs()>0
        assert 'weighted_specialization' not in values
    else:
        assert 'weighted_specialization' in values
        assert 'weighted_state_crps' not in values  # computed diagnostic, NOT A's objective
        assert 'weighted_transition_crps' not in values
    if stage=='C':
        assert values['weighted_loss_delta_member']==0
        assert 'weighted_marginal_crps' in values and 'weighted_pi' in values
        assert 'weighted_static_l2' not in values  # not implicitly adding A curriculum to C
    values['loss'].backward()
    modules={'representation':m.core.manifold,'information':m.information,
             'experts':m.core.experts,'gate':m.core.gate,'history':m.core.history_encoder,
             'auxiliary':m.a_sampler}
    for name,module in modules.items():
        active=(name in ('representation','information','auxiliary') if stage=='A' else
                name in ('experts','gate','history') if stage=='B' else name!='auxiliary')
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert bool(grads)==active,(stage,name)
        if active:
            assert all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads)>0
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    opt.step()
    if stage=='B':
        prefixes=('core.experts.','core.gate.correction.','core.history_encoder.')
        assert all(torch.equal(value,m.state_dict()[name]) for name,value in before.items()
                   if not name.startswith(prefixes))


def test_A_drift_only_skips_auxiliary_sampler(prepared,monkeypatch):
    m,b,_,_,_=prepared
    def forbidden(*args,**kwargs):raise AssertionError('drift-only evaluated stochastic sampler')
    monkeypatch.setattr(m,'auxiliary_field',forbidden)
    trace=[]
    path=m.rollout(b['history'],b['information'],auxiliary=True,drift_only=True,trace=trace)
    assert path.shape==(2,4,21,128)
    assert torch.equal(path[:,0],path[:,1])
    assert torch.equal(trace[1]['input'],trace[0]['output'])
    assert all(torch.count_nonzero(t['residual_per_day'])==0 for t in trace)
    path[:,:,-1].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.core.manifold.latent_drift.parameters())


def test_information_static_dynamic_labels_and_duplicate_options(prepared,tmp_path):
    _,_,d,archive,info=prepared
    with np.load(info) as f:payload={k:f[k] for k in f.files}
    meta=json.loads(str(payload['metadata_json']))
    meta['variables'][0]['kind']='unknown'
    payload['metadata_json']=json.dumps(meta)
    bad=tmp_path/'kind.npz';np.savez_compressed(bad,**payload)
    with pytest.raises(ValueError,match='Static/dynamic'):
        load_information(bad,archive,d['times'],d['schema'])
    with pytest.raises(ValueError,match='duplicate'):
        prepare(archive,tmp_path/'synthetic-information.nc',tmp_path/'duplicate.npz',optional=['t850','t850'])
    assert not (tmp_path/'duplicate.npz').exists()


def test_static_excluded_from_probabilistic_scores(prepared):
    m,b,d,_,_=prepared
    _,q=m.rollout(b['history'],b['information'],auxiliary=True,members=2,tau_steps=1,return_q=True)
    args=(q,b['information'],b['information_targets'],b['dt_hours'],
          torch.tensor(d['information_scale']),torch.tensor(d['information_tendency_scale']))
    before=m.information_scores(*args)
    changed=b['information_targets'].clone();changed[:,:,5*32:]+=9999
    after=m.information_scores(q,b['information'],changed,*args[3:])
    assert all(torch.equal(before[k],after[k]) for k in before)
    assert not any('terrain' in k for k in before)
    grad=torch.autograd.grad(before['info_distribution'],list(m.information.parameters()),retain_graph=True)
    assert sum(g.abs().sum() for g in grad)>0


def test_invalid_sampling_and_audit_guard(tmp_path):
    from climate_diffusion.train_information_process import main
    from audit_information_process import audit
    with pytest.raises(ValueError,match='max_windows'):
        main(['--archive','not-read.npz','--output',str(tmp_path/'new.pt'),'--stage','A','--max-windows','1'])
    with pytest.raises(ValueError,match='max_pairs'):
        audit('not-read.pt','not-read.npz',tmp_path/'audit.json',max_pairs=0)
    with pytest.raises(ValueError,match='never test'):
        audit('not-read.pt','not-read.npz',tmp_path/'audit.json',split='test')
