"""Ad-hoc sign all embedded Mach-O files before sealing the application."""
from pathlib import Path
import subprocess
import sys

app = Path(sys.argv[1]).resolve()
magic = {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
         b'\xfe\xed\xfa\xce', b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'}
count = 0
for path in sorted(app.rglob('*')):
    if path.is_symlink() or not path.is_file():
        continue
    with path.open('rb') as stream:
        if stream.read(4) not in magic:
            continue
    subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    count += 1
subprocess.run(['/usr/bin/codesign', '--force', '--sign', '-', str(app)], check=True)
subprocess.run(['/usr/bin/codesign', '--verify', '--deep', '--strict', str(app)], check=True)
print(f'Signed {count} embedded Mach-O files and verified {app.name}.')
