import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from stream_era5_extra import produce, ready, clean_raw, prune_verified_raw
from test_prepare_era5_extra import source
from smoke_moe import synthetic_archive
from climate_diffusion.physical_information import load_information, fit_information, digest, information_digest
from climate_diffusion.information_shards import InformationShards, publish_json
from climate_diffusion.moe_data import load_moe_archive, field_grid
from climate_diffusion.manifold_moe import ManifoldMoEConfig
from climate_diffusion.train_information_process import data_contract, Windows
from climate_diffusion.temporal_supervision import area_weights


class FakeCDS:
    def __init__(self):
        self.calls = 0

    def retrieve(self, dataset, request, output):
        self.calls += 1
        source(request, terrain=dataset.endswith('single-levels')).to_netcdf(output)


@pytest.fixture
def archive(tmp_path):
    torch.set_num_threads(1)
    path, _ = synthetic_archive(tmp_path, count=320)
    return path


def reader(store, archive):
    _, t, s = load_moe_archive(archive)
    return InformationShards(store, archive, t, s)


def test_verified_delete_resume_and_cross_chunk_rows(archive, tmp_path):
    client = FakeCDS(); store = tmp_path/'store'
    class Interrupted(Exception): pass
    def interrupt(i, _):
        if i == 1: raise Interrupted()
    with pytest.raises(Interrupted):
        produce(archive, store, client=client, delete_raw=True, on_commit=interrupt)
    assert not list((store/'raw').glob('*.nc'))
    first = digest(store/'chunks/000000.npz')
    r = reader(store, archive)
    assert r[10:15].shape == (5,224)
    np.testing.assert_array_equal(r[10:15], np.stack([r[i] for i in range(10,15)]))
    with pytest.raises(FileNotFoundError): _ = r[24]
    initial_calls = client.calls
    produce(archive, store, client=client, delete_raw=True)
    assert digest(store/'chunks/000000.npz') == first
    assert client.calls == initial_calls + 2*(len(r.ranges)-2)
    assert not list((store/'raw').glob('*.nc'))
    class NoNetwork:
        def retrieve(self, *a): pytest.fail('Completed shards must not re-download')
    identity = information_digest(store)
    produce(archive, store, client=NoNetwork(), delete_raw=True)
    assert information_digest(store) == identity


def test_corruption_preserves_raw_and_unowned_files(archive, tmp_path):
    store = tmp_path/'store'; client=FakeCDS()
    class Interrupted(Exception): pass
    def interrupt(i, _): raise Interrupted()
    with pytest.raises(Interrupted): produce(archive,store,client=client,on_commit=interrupt)
    raw = list((store/'raw').glob('*.nc'))
    # Keep terrain intact so failure is specifically the first dynamic shard.
    (store/'chunks/000000.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        produce(archive,store,client=client,delete_raw=True)
    assert sum(p.exists() for p in raw) >= 2  # dynamic inputs not removed
    other = tmp_path/'unowned'; other.mkdir(); (other/'mine.nc').write_text('retain')
    with pytest.raises(ValueError, match='nonempty'):
        produce(archive,other,client=client,delete_raw=True)
    assert (other/'mine.nc').read_text() == 'retain'


def test_orphan_after_atomic_publish_is_recovered_without_download(archive,tmp_path,monkeypatch):
    import stream_era5_extra as module
    real = module.publish_json
    def fail_receipt(path,value):
        if str(path).endswith('chunks/000000.json'): raise RuntimeError('power loss')
        return real(path,value)
    store=tmp_path/'store'; client=FakeCDS()
    monkeypatch.setattr(module,'publish_json',fail_receipt)
    with pytest.raises(RuntimeError,match='power loss'):
        produce(archive,store,client=client,delete_raw=True)
    assert (store/'chunks/000000.npz').exists()
    assert not (store/'chunks/000000.json').exists()
    with pytest.raises(FileNotFoundError): reader(store,archive)[0]
    monkeypatch.setattr(module,'publish_json',real)
    calls=client.calls
    produce(archive,store,client=client,delete_raw=True)
    assert client.calls == calls+2*(len(reader(store,archive).ranges)-1)


def test_partial_AB_stats_and_120h_windows_without_future_shards(archive,tmp_path):
    store=tmp_path/'store'
    class Ready(Exception): pass
    def stop(i, r):
        try: ready(store,archive,'AB',history_stride=1)
        except FileNotFoundError: return
        raise Ready()
    with pytest.raises(Ready): produce(archive,store,client=FakeCDS(),on_commit=stop)
    states,times,schema=load_moe_archive(archive)
    cfg=ManifoldMoEConfig(state_dim=128,grid=field_grid(schema),history_steps=6,history_stride=1,horizon_steps=20,step_hours=6)
    d=data_contract(archive,store,'enriched',cfg)
    raw=reader(store,archive)
    a=raw[:d['train_end']]
    im,sc=fit_information(a,raw.meta,len(a),schema)
    np.testing.assert_allclose(d['information_mean'],im,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(d['information_scale'],sc,atol=1e-6,rtol=1e-6)
    delta=np.diff(a.astype(np.float64),axis=0).reshape(-1,*raw.meta['shape'])/6
    w=area_weights(schema);mean=(delta*w).sum((-2,-1)).mean(0)
    sd=np.sqrt(((delta-mean[None,:,None,None])**2*w).sum((-2,-1)).mean(0))
    sd=np.maximum(sd,np.maximum(sc.reshape(7,4,8).mean((-2,-1))*1e-3/6,1e-8))
    np.testing.assert_allclose(d['information_tendency_scale'].reshape(7,4,8)[:,0,0],sd,rtol=1e-6)
    ds=Windows(states,times,cfg,[8],d['mean'],d['scale'],schema,information=d['information'])
    assert ds[0]['information_targets'].shape == (20,224)
    np.testing.assert_array_equal(ds[0]['information'].numpy(), d['information'][13])
    np.testing.assert_array_equal(ds[0]['information_targets'].numpy(),d['information'][14:34])
    assert not (store/'complete.json').exists()
    with pytest.raises(FileNotFoundError):ready(store,archive,'all')
    # Adding held-out data cannot alter frozen train statistics or dataset identity.
    identity=information_digest(store)
    produce(archive,store,client=FakeCDS(),delete_raw=True)
    e=data_contract(archive,store,'enriched',cfg)
    assert information_digest(store)==identity
    for key in ('mean','scale','information_mean','information_scale','information_tendency_scale'):
        np.testing.assert_array_equal(d[key],e[key])


@pytest.mark.parametrize('corrupt', ['mask','time','units','static'])
def test_chunk_contracts_fail_fast(archive,tmp_path,corrupt):
    store=tmp_path/'store'
    class Stop(Exception):pass
    def stop(i,r):raise Stop()
    with pytest.raises(Stop):produce(archive,store,client=FakeCDS(),on_commit=stop)
    r=reader(store,archive);p=r.chunk_path(0)
    with np.load(p) as f:content={k:f[k] for k in f.files}
    if corrupt=='mask':content['observed_mask'][0,0]=0
    elif corrupt=='time':content['times'][0]+=np.timedelta64(1,'h')
    elif corrupt=='static':content['data'][:,-1]+=1
    else:r.meta['variables'][0]['unit']='Pa'
    with open(p,'wb') as f:np.savez_compressed(f,**content)
    with pytest.raises(ValueError):r.validate_chunk(0)


def test_pins_symlinks_and_producer_lock(archive,tmp_path):
    import fcntl
    store=tmp_path/'store';store.mkdir()
    with open(store/'.producer.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        with pytest.raises(RuntimeError,match='Another producer'):produce(archive,store,client=FakeCDS())
    produce(archive,store,client=FakeCDS(),delete_raw=True)
    r=reader(store,archive);r[0];pinned=r.provenance()
    receipt=r.receipt_path(0);info=json.loads(receipt.read_text());info['sha256']='changed';receipt.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='Previously consumed'):reader(store,archive).pin(pinned)
    external=tmp_path/'outside.nc';external.write_text('do not delete')
    link=store/'raw/fake.nc';link.symlink_to(external)
    with pytest.raises(ValueError,match='symlink'):clean_raw(store/'raw',[{'name':link.name}])
    assert external.read_text()=='do not delete'
    with pytest.raises(ValueError,match='plan changed'):produce(archive,store,days=2,client=FakeCDS())


def test_final_reconciliation_revalidates_before_removing_owned_sources(archive,tmp_path):
    store=tmp_path/'store'
    produce(archive,store,client=FakeCDS(),delete_raw=False)
    assert len(list((store/'raw').glob('*.nc')))>1
    assert prune_verified_raw(store,archive)['remaining_nc']==0
    assert prune_verified_raw(store,archive)['remaining_nc']==0
    # Changed converted bytes forbid reconciliation, even if a receipt exists.
    (store/'chunks/000000.npz').write_bytes(b'bad')
    with pytest.raises(ValueError,match='checksum'):prune_verified_raw(store,archive)


def test_shell_runner_overlap_order_and_no_overwrite(tmp_path):
    """Run the real shell runners, mocking only costly Python/CDS commands."""
    import os
    import subprocess
    helper=tmp_path/'fake-python'
    helper.write_text('''#!/usr/bin/env python3
import os,sys,time,json
from pathlib import Path
a=sys.argv[1:]; run=Path(os.environ['RUN']); info=Path(os.environ['INFO'])
def value(k):return a[a.index(k)+1]
with open(run/'calls.jsonl','a') as f:f.write(json.dumps(a)+'\\n')
if a[0].endswith('stream_era5_extra.py'):
    if '--download' in a:
        info.mkdir(exist_ok=True); (info/'prefix').touch()
        deadline=time.monotonic()+15
        while not (run/'B-done').exists():
            if time.monotonic()>deadline:sys.exit(9)
            time.sleep(.05)
        (info/'all').touch()
    elif '--check-ready' in a:
        ok=(info/('prefix' if value('--check-ready')=='AB' else 'all')).exists()
        print(json.dumps({'ready':ok})); sys.exit(0 if ok else 75)
    else: print('{}')
elif '-m' in a and value('-m')=='climate_diffusion.train_information_process':
    stage=value('--stage')
    if stage=='A':assert (info/'prefix').exists() and not (info/'all').exists()
    if stage=='C':assert (info/'all').exists()
    (run/(stage+'-done')).touch()
''')
    helper.chmod(0o755)
    env={**os.environ,'PYTHON':str(helper),'ARCHIVE':str(tmp_path/'surface.npz'),
         'INFO':str(tmp_path/'shards'),'RUN':str(tmp_path/'run'),'THROUGH':'render'}
    script=Path(__file__).resolve().parents[1]/'scripts/run_streaming_a_information.sh'
    proc=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=25)
    assert proc.returncode==0,proc.stdout+proc.stderr
    calls=[json.loads(x) for x in (tmp_path/'run/calls.jsonl').read_text().splitlines()]
    training=[a[a.index('--stage')+1] for a in calls if '--stage' in a]
    assert training==['A','B','C']
    rendering=[a[a.index('--interval-hours')+1] for a in calls if '--interval-hours' in a]
    assert rendering==['6','12']
    assert calls[-1][-1]=='--prune-verified-raw'
    again=subprocess.run(['bash',str(script)],env=env,capture_output=True,text=True,timeout=5)
    assert again.returncode!=0 and 'new RUN' in again.stderr
