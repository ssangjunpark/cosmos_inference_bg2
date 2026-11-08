#!/usr/bin/env python3
"""Thin 5090 gateway: FFW observations → persistent inference API → FFW chunk inbox."""
import argparse
from pathlib import Path
import json
import signal
import statistics
import time
import uuid

from bg2_bridge.core import require, validate_profile
from bg2_bridge.gateway import Gateway, GatewayTransport, StatusCache, DebugGateway
from bg2_bridge.observations import Observations
from bg2_bridge.rpc import InferenceClient
from bg2_inference import load_observation
from bg2_bridge.recording import RunRecorder

ROOT=Path(__file__).resolve().parents[1]


def finish_control(transport,status,command):
    """Best-effort control with an observed acknowledgement before closing transport."""
    current=status.get(time.monotonic_ns())
    if current is None or current[0]['mode']!='execute':
        return {'event':'control_not_confirmed','command':command,'reason':'No fresh FFW status'}
    if command=='pause' and current[0]['phase']=='paused':
        return {'event':'pause_confirmed','epoch':current[0]['epoch']}
    if current[0]['phase'] not in {'ready','running','holding','paused'}:
        return {'event':'control_not_sent','command':command,'phase':current[0]['phase']}
    epoch=current[0]['epoch']
    transport.control(epoch,command)
    deadline=time.monotonic()+1.
    wanted='paused' if command=='pause' else 'stopped'
    while time.monotonic()<deadline:
        latest=status.get(time.monotonic_ns())
        if latest and latest[0]['phase']==wanted and latest[0]['epoch']!=epoch:
            return {'event':command+'_confirmed','epoch':latest[0]['epoch']}
        time.sleep(.01)
    return {'event':'control_not_confirmed','command':command,'reason':'No FFW acknowledgement; check robot status'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','preview','execute','replay','collect-recordings'))
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/robotis_bg2_rev4.reference.json')
    parser.add_argument('--server-url',default='http://127.0.0.1:8008')
    parser.add_argument('--messages-dir',type=Path,default=ROOT/'deploy/gateway/messages')
    parser.add_argument('--instruction',default='')
    parser.add_argument('--timeout',type=float,default=10.)
    parser.add_argument('--duration',type=float,default=0.,help='Zero runs until interrupted')
    parser.add_argument('--replay-observation',type=Path)
    parser.add_argument('--requests',type=int,default=5,help='Number of recorded-input replay requests')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--debug',action='store_true',help='Live predictions and recordings only; no FFW command publishers or homing requirement')
    parser.add_argument('--max-predictions',type=int,default=0,help='Debug only: stop after this many predictions; zero runs until Ctrl+C')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--recording-wait',type=float,default=30.,help='Seconds to collect pending imagined clips after control stops; Ctrl+C again skips this wait')
    args=parser.parse_args(argv)
    require(args.duration>=0 and args.requests>0 and args.max_predictions>=0,'Invalid duration or count')
    require(args.recording_wait>=0,'Invalid recording wait')
    require(not args.debug or args.mode in ('preview','execute'),'Use --debug with preview or execute')
    require(args.debug or args.max_predictions==0,'--max-predictions is for --debug only')
    if args.mode=='collect-recordings':
        from bg2_bridge.rollouts import RunRollouts
        require((args.output/'imagined/manifest.json').is_file() or (args.output/'predictions').is_dir(),
                'No prediction records or imagined manifest in this run')
        collector=RunRollouts(args.output,InferenceClient(args.server_url,5),resume=True)
        signal.signal(signal.SIGINT,lambda *_:collector.cancel())
        signal.signal(signal.SIGTERM,lambda *_:collector.cancel())
        collector.close(wait=args.recording_wait)
        return
    profile=json.loads(args.profile.read_text())
    validate_profile(profile,live=args.mode in ('preview','execute'),execute=args.mode=='execute' and not args.debug)
    client=None
    if args.mode!='inspect':
        client=InferenceClient(args.server_url,args.timeout,run_id=args.output.name)
    if args.mode in ('preview','execute'):require(args.instruction.strip(),'Supply an instruction')
    if args.mode=='replay':require(args.replay_observation is not None,'Supply a recorded observation')
    if args.check:
        print(json.dumps({'valid':True,'mode':'debug' if args.debug else args.mode,'physical_publishing':args.mode=='execute' and not args.debug,'robot_connection_opened':False}));return
    args.output.mkdir(parents=True,exist_ok=False)
    health=client.health() if client is not None else None
    if args.mode=='replay':
        from bg2_bridge.artifact_protocol import ARTIFACT_API
        from bg2_bridge.rollouts import RunRollouts
        collector=RunRollouts(args.output,client.recording_client()) if health.get('artifacts_api')==ARTIFACT_API else None
        observation=load_observation(args.replay_observation)
        elapsed=[];last=None
        try:
            for _ in range(args.requests):
                start=time.monotonic();epoch,request_id=uuid.uuid4().hex,uuid.uuid4().hex
                if collector is not None:collector.register(epoch,request_id,int(observation['source_time_ns']))
                last=client.predict(observation,epoch,request_id)
                if collector is not None:collector.prediction(last)
                elapsed.append(time.monotonic()-start)
        finally:
            if collector is not None:collector.close(wait=args.recording_wait)
        (args.output/'last_prediction.json').write_text(json.dumps(last,indent=2)+'\n')
        report={'mode':'recorded_replay','physical_publishing':False,'requests':args.requests,'health':health,
                'roundtrip_seconds':elapsed,'median_seconds':statistics.median(elapsed),'max_seconds':max(elapsed),
                'original_source_time_ns':int(observation['source_time_ns'])}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    if client is not None:require(health.get('records_outputs') is True,'Restart the PRO model service with output recording enabled')
    from bg2_bridge.artifact_protocol import ARTIFACT_API
    recorder=RunRecorder(args.output,'debug' if args.debug else args.mode,health=health,
        artifact_client=client.recording_client() if health.get('artifacts_api')==ARTIFACT_API else None,
        recording_wait=args.recording_wait) if client is not None else None
    observations,status=Observations(profile),StatusCache()
    transport=None
    engine=None
    stopped=False
    def stop(*_):
        nonlocal stopped
        if stopped and recorder is not None:recorder.cancel_collection()
        stopped=True
        if engine is not None: engine.close()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();last_report=0.;last_event=None;normal_exit=False
    try:
        transport=GatewayTransport(profile,observations,status,inspect=args.mode=='inspect' or args.debug,messages_dir=args.messages_dir,recorder=recorder)
        if args.debug:
            engine=DebugGateway(observations,client,recorder,args.instruction)
        elif client is not None:
            engine=Gateway(observations,status,client,transport.send_chunk,args.instruction,args.mode,recorder=recorder,send_control=transport.control)
        with (args.output/'events.jsonl').open('x') as journal:
            if health:journal.write(json.dumps({'event':'server_ready','health':health})+'\n')
            while not stopped and (args.duration==0 or time.monotonic()-start<args.duration):
                now=time.monotonic_ns()
                event=engine.step(now) if engine is not None else None
                if event and (event['event']!='waiting' or time.monotonic()-last_report>=1):
                    journal.write(json.dumps(event,allow_nan=False)+'\n');journal.flush();last_event=event
                if args.debug and args.max_predictions and engine.predictions>=args.max_predictions:break
                if time.monotonic()-last_report>=1:
                    summary={'event':'observations','data':transport.diagnostics(time.monotonic_ns()),'gateway_event':last_event}
                    journal.write(json.dumps(summary,allow_nan=False)+'\n');journal.flush()
                    print(json.dumps(summary),flush=True);last_report=time.monotonic()
                time.sleep(.01)
        normal_exit=True
    finally:
        # Ctrl+C/duration expiry pause; unexpected failures retain a terminal stop.
        if transport is not None and engine is not None and not args.debug and args.mode=='execute':
            try:
                event=finish_control(transport,status,'pause' if normal_exit else 'stop')
                print(json.dumps(event),flush=True)
                with (args.output/'events.jsonl').open('a') as journal:journal.write(json.dumps(event)+'\n')
            except Exception as error:
                print(json.dumps({'event':'control_not_confirmed','reason':str(error)}),flush=True)
        if engine is not None:engine.close()
        try:
            if transport is not None:transport.close()
        finally:
            if recorder is not None:recorder.close()


if __name__=='__main__':
    main()
