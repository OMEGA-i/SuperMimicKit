import pytest
from scripts.swanlab_tracking import split_metrics, checkpoint_asset_env


def test_asset_migration_preserves_historical_physics():
    cfg={'source_asset_sha256':'old','asset_env':{'MODEL':'old.xml'},
         'asset_versions':{'new':{'MODEL':'new.xml'}}}
    assert checkpoint_asset_env(cfg,'old') == {'MODEL':'old.xml'}
    assert checkpoint_asset_env(cfg,'new') == {'MODEL':'new.xml'}
    with pytest.raises(ValueError):checkpoint_asset_env(cfg,'unknown')


def test_independent_split_means_and_failures():
    manifest={'splits':{'train':[f't{i}' for i in range(40)],'test':[f'v{i}' for i in range(37)]}}
    rows=[{'motion':n,'normalized_return':.8,'failed':False} for n in manifest['splits']['train']]
    rows += [{'motion':n,'normalized_return':.2,'failed':True} for n in manifest['splits']['test']]
    result={'checkpoint_iterations':500,'motions':rows}
    groups=split_metrics(result,manifest)
    assert groups['train40']['normalized_return']==pytest.approx(.8)
    assert groups['val37']['failed_motions']==37
    assert groups['val37']['iteration']==500
    result['motions']=rows[:-1]
    with pytest.raises(ValueError,match='expected 37'):split_metrics(result,manifest)
