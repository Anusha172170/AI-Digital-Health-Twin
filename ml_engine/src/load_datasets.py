"""Download pinned public benchmark mirrors, retaining exact source bytes."""
import base64
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
import requests
from ml_engine.src.config import RAW, PIMA_COLUMNS

SOURCES = {
    'diabetes': ('jbrownlee/Datasets', 'd20fcb6402ae34e653d4513b00f39257bb37ed7f', 'pima-indians-diabetes.data.csv'),
    'framingham': ('OpenMined/TenSEAL', 'c962074f0aa82cf30eeaddf17af887a8ea0e66fe', 'tutorials/data/framingham.csv'),
}

def download():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, commit, path) in SOURCES.items():
        url = f'https://raw.githubusercontent.com/{repo}/{commit}/{path}'
        transport = 'https'
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            content = response.content
        except requests.RequestException:
            # GitHub API transport for environments that block raw.githubusercontent.com.
            transport = 'github_contents_api'
            result = subprocess.run(['gh', 'api', f'repos/{repo}/contents/{path}?ref={commit}'],
                                    check=True, capture_output=True, text=True)
            content = base64.b64decode(json.loads(result.stdout)['content'])
        df = pd.read_csv(io.BytesIO(content), **({'names': PIMA_COLUMNS} if name == 'diabetes' else {}))
        required = PIMA_COLUMNS if name == 'diabetes' else ['TenYearCHD', 'prevalentHyp', 'sysBP', 'BMI']
        if not set(required).issubset(df.columns) or len(df) < 700:
            raise ValueError(f'Unexpected schema or record count for {name}')
        original = RAW / f'{name}_source.csv'
        original.write_bytes(content)
        df.to_csv(RAW / f'{name}.csv', index=False)
        manifest[name] = {'url': url, 'mirror_repository': repo, 'commit': commit,
            'transport': transport, 'sha256': hashlib.sha256(content).hexdigest(),
            'downloaded_at': datetime.now(timezone.utc).isoformat(), 'rows': len(df),
            'columns': list(df.columns), 'source_file': original.name,
            'note': 'Public benchmark mirror, not a direct download from original clinical institution. No synthetic rows. Upstream dataset rights require review before redistribution beyond this research repository.'}
        print(f'{name}: {len(df)} rows, saved exact source and named-column CSV')
    (RAW / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    download()
