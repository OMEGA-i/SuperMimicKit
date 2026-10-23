import torch
from supermimic.checkpoints import save_evaluation_checkpoint


def test_paired_archive_excludes_replay_and_is_immutable(tmp_path):
    d={key:{} for key in ['config','args','training_contract']}
    d.update(joint_names=['joint'],body_names=['body'],iteration=9999,learner_version=2,
             actor={'weight':torch.ones(2)},world={'weight':torch.full((3,),2.)},
             replay={'large_buffer':torch.zeros(10)},optimizers=['not retained'])
    p=tmp_path/'ckpt_10000.pt'
    assert save_evaluation_checkpoint(d,p,include_world=True)
    d['world']['weight'].fill_(9)
    assert not save_evaluation_checkpoint(d,p,include_world=True)
    saved=torch.load(p,weights_only=False)
    assert set(saved)=={'config','args','training_contract','joint_names','body_names','iteration','learner_version','actor','world'}
    assert saved['world']['weight'].tolist()==[2.,2.,2.]
    q=tmp_path/'eval_10000.pt'
    assert save_evaluation_checkpoint(d,q)
    assert 'world' not in torch.load(q,weights_only=False)
