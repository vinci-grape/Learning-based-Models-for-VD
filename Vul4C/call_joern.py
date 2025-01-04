
import subprocess
import time
import os
import argparse
import sys
import signal

def run_joern(working_dir,timeout):
    try:
        p = subprocess.Popen('sbt "testOnly io.joern.c2cpg.io.Vul4CTest"',shell=True, start_new_session=True,cwd=working_dir)
        p.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        print(f'Timeout for joern expired', file=sys.stderr)
        print('Terminating the whole process group...', file=sys.stderr)
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        time.sleep(10)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-sbt",required=True)
    parser.add_argument("-java",required=True)
    parser.add_argument("-scala",required=True)
    parser.add_argument("-working_dir",required=True)
    parser.add_argument("-timeout",default=20,type=int)
    args = parser.parse_args()
    print('java',args.java)
    print('scala',args.scala)
    print('sbt',args.sbt)
    os.environ['PATH'] += ':'+args.java
    os.environ['PATH'] += ':'+args.scala
    os.environ['PATH'] += ':'+args.sbt
    print(os.environ['PATH'])
    cnt = 1
    print(f'[RUNNING] run joern [{cnt}] times')
    while not run_joern(args.working_dir, args.timeout):
        cnt +=1
        print(f'[RUNNING] run joern [{cnt}] times')
    print('completed!!!\n'*3)
    sys.exit(0)
