#!/usr/bin/env python3
"""Thin 5090 gateway: FFW observations → persistent inference API → FFW chunk inbox."""
import argparse
from pathlib import Path
import json
import signal
import statistics
import threading
import time
import uuid

from bg2_bridge.core import require, validate_profile
from bg2_bridge.gateway import Gateway, GatewayTransport, StatusCache, DebugGateway
from bg2_bridge.observations import Observations
from bg2_bridge.rpc import InferenceClient
from bg2_inference import load_observation
from bg2_bridge.recording import CameraRecorder, RunRecorder

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
    parser.add_argument('mode',choices=('inspect','preview','execute','replay','collect-recordings','record-cameras'))
    parser.add_argument('--profile',type=Path,default=ROOT/'profiles/robotis_bg2_rev4.reference.json')
    parser.add_argument('--server-url',default='http://127.0.0.1:8008')
    parser.add_argument('--expected-model', help='Require the server health model_id before opening robot transport')
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
    parser.add_argument('--pause-diagnostic',action='store_true',
                        help='After Ctrl+C and a confirmed pause, save one fresh WAM forecast without executing it')
    parser.add_argument('--quiet-status',action='store_true',help='Keep detailed camera status in events.jsonl without printing it')
    args=parser.parse_args(argv)
    require(args.duration>=0 and args.requests>0 and args.max_predictions>=0,'Invalid duration or count')
    require(args.recording_wait>=0,'Invalid recording wait')
    require(not args.debug or args.mode in ('preview','execute'),'Use --debug with preview or execute')
    require(args.debug or args.max_predictions==0,'--max-predictions is for --debug only')
    require(args.expected_model is None or args.mode in ('preview','execute','replay'),
            '--expected-model requires a prediction mode')
    require(not args.pause_diagnostic or (args.mode=='execute' and not args.debug),
            '--pause-diagnostic is only for normal execute mode')
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
    validate_profile(profile,live=args.mode in ('preview','execute','record-cameras'),execute=args.mode=='execute' and not args.debug)
    client=None
    if args.mode in ('preview','execute','replay'):
        client=InferenceClient(args.server_url,args.timeout,run_id=args.output.name)
    if args.mode in ('preview','execute'):require(args.instruction.strip(),'Supply an instruction')
    if args.mode=='replay':require(args.replay_observation is not None,'Supply a recorded observation')
    if args.check:
        print(json.dumps({'valid':True,'mode':'debug' if args.debug else args.mode,'physical_publishing':args.mode=='execute' and not args.debug,'robot_connection_opened':False,
                          'pause_diagnostic':args.pause_diagnostic}));return
    args.output.mkdir(parents=True,exist_ok=False)
    health=client.health() if client is not None else None
    if args.expected_model is not None:
        require(health.get('model_id')==args.expected_model,
                f"Expected model {args.expected_model}, server reports {health.get('model_id')}")
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
    from bg2_bridge.artifact_protocol import ARTIFACT_API, atomic_json
    if args.pause_diagnostic:
        require(health.get('mode')=='wam' and health.get('artifacts_api')==ARTIFACT_API,
                '--pause-diagnostic requires a WAM server with recording enabled')
    if args.mode=='record-cameras':
        recorder=CameraRecorder(args.output,profile)
    else:
        recorder=RunRecorder(args.output,'debug' if args.debug else args.mode,health=health,
            artifact_client=client.recording_client() if health.get('artifacts_api')==ARTIFACT_API else None,
            recording_wait=args.recording_wait) if client is not None else None
    if client is not None:
        recorder.metadata['pause_diagnostic_enabled']=args.pause_diagnostic
        atomic_json(args.output/'run.json',recorder.metadata)
    observations,status=Observations(profile),StatusCache()
    transport=None
    engine=None
    stopped=False
    stop_signal=None
    diagnostic=None
    diagnostic_result=None
    diagnostic_abort=threading.Event()
    def stop(signum=signal.SIGINT,*_):
        nonlocal stopped,stop_signal
        if stopped:
            diagnostic_abort.set()
            if diagnostic is not None:diagnostic.cancel()
            if recorder is not None:recorder.cancel_collection()
        else:stop_signal=signum
        stopped=True
        if engine is not None: engine.close()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    start=time.monotonic();last_report=0.;last_event=None;normal_exit=False
    try:
        transport_options={'inspect':args.mode in ('inspect','record-cameras') or args.debug,
                           'messages_dir':args.messages_dir,'recorder':recorder}
        if args.mode=='record-cameras':transport_options['cameras_only']=True
        transport=GatewayTransport(profile,observations,status,**transport_options)
        if args.debug:
            engine=DebugGateway(observations,client,recorder,args.instruction)
        elif client is not None:
            engine=Gateway(observations,status,client,transport.send_chunk,args.instruction,args.mode,recorder=recorder,send_control=transport.control)
        with (args.output/'events.jsonl').open('x') as journal:
            if health:journal.write(json.dumps({'event':'server_ready','health':health})+'\n')
            while not stopped and (args.duration==0 or time.monotonic()-start<args.duration):
                if args.mode=='record-cameras':recorder.check()
                now=time.monotonic_ns()
                event=engine.step(now) if engine is not None else None
                if event and (event['event']!='waiting' or time.monotonic()-last_report>=1):
                    journal.write(json.dumps(event,allow_nan=False)+'\n');journal.flush();last_event=event
                if args.debug and args.max_predictions and engine.predictions>=args.max_predictions:break
                if time.monotonic()-last_report>=1:
                    summary={'event':'observations','data':transport.diagnostics(time.monotonic_ns()),'gateway_event':last_event}
                    journal.write(json.dumps(summary,allow_nan=False)+'\n');journal.flush()
                    if not args.quiet_status:print(json.dumps(summary),flush=True)
                    last_report=time.monotonic()
                time.sleep(.01)
        normal_exit=True
    finally:
        # Ctrl+C/duration expiry pause; unexpected failures retain a terminal stop.
        pause_event=None
        if transport is not None and engine is not None and not args.debug and args.mode=='execute':
            try:
                event=finish_control(transport,status,'pause' if normal_exit else 'stop')
                pause_event=event
                print(json.dumps(event),flush=True)
                with (args.output/'events.jsonl').open('a') as journal:journal.write(json.dumps(event)+'\n')
            except Exception as error:
                print(json.dumps({'event':'control_not_confirmed','reason':str(error)}),flush=True)
        if engine is not None:engine.close()
        if args.pause_diagnostic and normal_exit and stop_signal==signal.SIGINT:
            if pause_event is not None and pause_event.get('event')=='pause_confirmed':
                try:
                    from bg2_bridge.pause_diagnostic import PauseDiagnostic
                    print(json.dumps({'event':'pause_diagnostic_started','submitted_to_ffw':False}),flush=True)
                    diagnostic=PauseDiagnostic(args.output,observations,status,
                        InferenceClient(args.server_url,min(args.timeout,5),run_id=args.output.name),
                        args.instruction,pause_event['epoch'],expected_model=args.expected_model,
                        cancelled=diagnostic_abort,prediction_timeout=args.timeout,recording_wait=args.recording_wait)
                    result=diagnostic.run()
                    diagnostic_result=result
                    recorder.metadata['pause_diagnostic']=result
                    if result['state']=='forecast_saved':recorder.confirm_inference_barrier(result)
                    event={'event':'pause_diagnostic_forecast_saved' if result['state']=='forecast_saved' else 'pause_diagnostic_incomplete',
                           'state':result['state'],'submitted_to_ffw':False,
                           'directory':str(args.output/'diagnostics/pause'),'error':result.get('error')}
                    print(json.dumps(event),flush=True)
                    with (args.output/'events.jsonl').open('a') as journal:journal.write(json.dumps(event)+'\n')
                except Exception as error:
                    print(json.dumps({'event':'pause_diagnostic_failed','error':str(error),'submitted_to_ffw':False}),flush=True)
            else:
                print(json.dumps({'event':'pause_diagnostic_skipped','reason':'Robot pause was not confirmed'}),flush=True)
        try:
            if transport is not None:transport.close()
        finally:
            if recorder is not None:recorder.close()
            if diagnostic is not None and diagnostic_result is not None and diagnostic_result['state']=='forecast_saved':
                result=diagnostic.finalize()
                recorder.metadata['pause_diagnostic']=result
                run_meta=json.loads((args.output/'run.json').read_text())
                run_meta['pause_diagnostic']=result
                atomic_json(args.output/'run.json',run_meta)
                event={'event':'pause_diagnostic_saved' if result['state']=='complete' else 'pause_diagnostic_incomplete',
                       'state':result['state'],'submitted_to_ffw':False,
                       'directory':str(args.output/'diagnostics/pause'),'error':result.get('error')}
                print(json.dumps(event),flush=True)
                with (args.output/'events.jsonl').open('a') as journal:journal.write(json.dumps(event)+'\n')


if __name__=='__main__':
    main()
