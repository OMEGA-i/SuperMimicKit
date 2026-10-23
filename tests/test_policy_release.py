import json
from pathlib import Path
import pytest
import torch
pytest.importorskip('onnx')
pytest.importorskip('onnxruntime')
from scripts.export_policy_release import export


@pytest.mark.parametrize('iteration,robot,version', [(49998,'g1',2),(49999,'a3',2),(49999,'g1',1)])
def test_release_rejects_wrong_iteration_config_or_version(tmp_path,iteration,robot,version):
    cfg=json.loads(Path('configs/g1.json').read_text())
    ckpt=tmp_path/'checkpoint.pt'
    torch.save({'iteration':iteration,'config':cfg,'learner_version':version},ckpt)
    output=tmp_path/'release'
    with pytest.raises(ValueError):export(ckpt,output,robot,50000)
    assert not output.exists()
