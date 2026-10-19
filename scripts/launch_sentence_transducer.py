"""Launch one ongoing training run independently of the observing tool session."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extra',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--embedding-scale',type=float)
    args=parser.parse_args()
    output=args.output.resolve()
    if args.resume:
        if not (output/'recovery.pt').is_file():raise ValueError('no recovery checkpoint')
    elif output.exists():
        raise FileExistsError('preserve the previous run; choose a new output directory')
    suffix='-resume-'+uuid.uuid4().hex[:12] if args.resume else ''
    receipt=output.with_name(output.name+suffix+'.launch.json')
    log=output.with_name(output.name+suffix+'.log')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive launch receipt prevents two fresh launches for the same output.
    with receipt.open('x') as record, log.open('x') as stream:
        command=[sys.executable,str(ROOT/'scripts/train_sentence_transducer.py'),
                 '--output',str(output),'--extra',str(args.extra),
                 '--qualification',str(args.qualification),'--epochs',str(args.epochs)]
        if args.resume:command.append('--resume')
        if args.embedding_scale is not None:
            command.extend(['--embedding-scale',str(args.embedding_scale)])
        environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                                 stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,started_at=datetime.now(timezone.utc).isoformat(),
                   command=command,output=str(output),log=str(log),resume=args.resume,
                   trainer_sha256=hashlib.sha256((ROOT/'scripts/train_sentence_transducer.py').read_bytes()).hexdigest(),
                   qualification_sha256=hashlib.sha256(args.qualification.read_bytes()).hexdigest())
        json.dump(value,record,indent=2);record.write('\n');record.flush();os.fsync(record.fileno())
    print(json.dumps(value),flush=True)


if __name__=='__main__':
    main()
