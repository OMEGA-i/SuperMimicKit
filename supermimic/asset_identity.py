"""Verify path-only relocation without weakening checkpoint physics identity."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(path):
    path = Path(path)
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    settings = dict(compiler.attrib) if compiler is not None else {}
    for kind, key in [('mesh', 'meshdir'), ('texture', 'texturedir')]:
        base = path.parent / settings.get(key, settings.get('assetdir', ''))
        for item in root.findall('./asset/' + kind):
            if 'file' in item.attrib:
                item.set('file', digest(base / item.get('file')))
    if compiler is not None:
        for key in ('meshdir', 'texturedir', 'assetdir'):
            compiler.attrib.pop(key, None)
    def canonical(node):
        return (node.tag, tuple(sorted(node.attrib.items())),
                (node.text or '').strip(), tuple(canonical(c) for c in node))
    return canonical(root)


def verify_asset(path, expected_hash):
    """Accept exact XML or a packaged, byte-verified path-only relocation."""
    path = Path(path)
    if digest(path) == expected_hash:
        return
    proof = json.loads((path.parent / 'provenance.json').read_text())
    source = Path(proof['source_xml'])
    if proof['source_xml_sha256'] != expected_hash or digest(source) != expected_hash:
        raise ValueError('Relocation source does not match checkpoint')
    if proof['xml_sha256'] != digest(path):
        raise ValueError('Packaged XML changed after validation')
    for asset in proof['assets']:
        if digest(path.parent / asset['file']) != asset['sha256']:
            raise ValueError('Packaged mesh/texture changed after validation')
    if signature(source) != signature(path):
        raise ValueError('Asset relocation changed model contents')
