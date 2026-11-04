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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('inspect','preview','execute','replay'))
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
    args=parser.parse_args(argv)
    require(args.duration>=0 and args.requests>0 and args.max_predictions>=0,'Invalid duration or count')
    require(not args.debug or args.mode in ('preview','execute'),'Use --debug with preview or execute')
    require(args.debug or args.max_predictions==0,'--max-predictions is for --debug only')
    profile=json.loads(args.profile.read_text())
    validate_profile(profile,live=args.mode in ('preview','execute'),execute=args.mode=='execute' and not args.debug)
    client=None
    if args.mode!='inspect':
        client=InferenceClient(args.server_url,args.timeout)
    if args.mode in ('preview','execute'):require(args.instruction.strip(),'Supply an instruction')
    if args.mode=='replay':require(args.replay_observation is not None,'Supply a recorded observation')
    if args.check:
        print(json.dumps({'valid':True,'mode':'debug' if args.debug else args.mode,'physical_publishing':args.mode=='execute' and not args.debug,'robot_connection_opened':False}));return
    args.output.mkdir(parents=True,exist_ok=False)
    health=client.health() if client is not None else None
    if args.mode=='replay':
        observation=load_observation(args.replay_observation)
        elapsed=[];last=None
        for _ in range(args.requests):
            start=time.monotonic()
            last=client.predict(observation,uuid.uuid4().hex,uuid.uuid4().hex)
            elapsed.append(time.monotonic()-start)
        (args.output/'last_prediction.json').write_text(json.dumps(last,indent=2)+'\n')
        report={'mode':'recorded_replay','physical_publishing':False,'requests':args.requests,'health':health,
                'roundtrip_seconds':elapsed,'median_seconds':statistics.median(elapsed),'max_seconds':max(elapsed),
                'original_source_time_ns':int(observation['source_time_ns'])}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    if client is not None:require(health.get('records_outputs') is True,'Restart the PRO model service with output recording enabled')
    recorder=RunRecorder(args.output,'debug' if args.debug else args.mode) if client is not None else None
    observations,status=Observations(profile),StatusCache()
    transport=None
    engine=None
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
        if engine is not None: engine.close()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();last_report=0.;last_event=None
    try:
        transport=GatewayTransport(profile,observations,status,inspect=args.mode=='inspect' or args.debug,messages_dir=args.messages_dir,recorder=recorder)
        if args.debug:
            engine=DebugGateway(observations,client,recorder,args.instruction)
        elif client is not None:
            engine=Gateway(observations,status,client,transport.send_chunk,args.instruction,args.mode,recorder=recorder)
        with (args.output/'events.jsonl').open('x') as journal:
            if health:journal.write(json.dumps({'event':'server_ready','health':health})+'\n')
            while not stopped and (args.duration==0 or time.monotonic()-start<args.duration):
                now=time.monotonic_ns()
                event=engine.step(now) if engine is not None else None
                if event and (event['event']!='waiting' or time.monotonic()-last_report>=1):
                    journal.write(json.dumps(event,allow_nan=False)+'\n');journal.flush();last_event=event
                if args.debug and args.max_predictions and engine.predictions>=args.max_predictions:break
                if time.monotonic()-last_report>=1:
                    summary={'event':'observations','data':transport.diagnostics(now),'gateway_event':last_event}
                    journal.write(json.dumps(summary,allow_nan=False)+'\n');journal.flush()
                    print(json.dumps(summary),flush=True);last_report=time.monotonic()
                time.sleep(.01)
    finally:
        # A requested stop is best effort; the FFW's independent final-command hold remains
        # responsible for network-loss behavior. No joint command is sent here.
        current=status.get(time.monotonic_ns())
        if transport is not None and engine is not None and not args.debug and args.mode=='execute' and engine.forwarded and current and current[0]['mode']=='execute':
            try:
                transport.stop(current[0]['epoch'])
            except Exception as error:
                print(json.dumps({'event':'stop_not_delivered','reason':str(error)}),flush=True)
        if engine is not None:engine.close()
        try:
            if transport is not None:transport.close()
        finally:
            if recorder is not None:recorder.close()


if __name__=='__main__':
    main()
