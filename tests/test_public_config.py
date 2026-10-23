"""Keep public robot entrypoints consistent across configuration and demo UI."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_public_robot_registry_and_files():
    robots = json.loads((ROOT / 'configs/robots.json').read_text())
    assert set(robots) == {'g1', 'h2', 'a3'}
    for name, cfg in robots.items():
        assert (ROOT / f'configs/{name}.json').is_file()
        assert (ROOT / cfg['pipeline_config']).is_file()
        assert (ROOT / cfg['export_config']).is_file()
        assert (ROOT / f'assets/robots/{name}/README.md').is_file()
    html = (ROOT / 'web/demo/dist/index.html').read_text()
    assert set(re.findall(r'<option value="([^"]+)"', html)) == set(robots)
