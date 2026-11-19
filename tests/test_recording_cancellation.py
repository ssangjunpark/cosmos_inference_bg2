"""Interrupted requests do not create phantom missing videos or hide real losses."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]

import gateway
from bg2_bridge.rollouts import RunRollouts
from bg2_bridge.rpc import InferenceClient,InferenceNotSubmitted,RecordingNotAvailable
from test_gateway import profile


class Client:
    url='http://127.0.0.1:18008'
    available=False
    def recording_status(self,path):
        if not self.available:raise RecordingNotAvailable('Recording not available')
        _,_,run,epoch,request_id=path.split('/')
        return {'state':'complete','run_id':run,'epoch':epoch,'request_id':request_id,'files':{}}
    def cancel(self):pass


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)/'run';self.root.mkdir()
        self.client=Client();self.epoch=uuid.uuid4().hex;self.request=uuid.uuid4().hex
        self.collector=RunRollouts(self.root,self.client)
        self.addCleanup(self.close)

    def close(self):
        if not self.collector.owner.closed:
            with mock.patch('bg2_bridge.rollouts.concatenate'),redirect_stdout(StringIO()):
                self.collector.close(wait=0)

    def barrier(self):
        return {'state':'forecast_saved','run_id':self.root.name,'server_url':self.client.url,
                'request_id':uuid.uuid4().hex,'pause_epoch':uuid.uuid4().hex,
                'source_time_ns':200,'submitted_to_ffw':False,'executed':False}

    def wait(self,condition):
        deadline=time.monotonic()+2
        while not condition():
            if time.monotonic()>deadline:self.fail('Collector did not reach expected state')
            self.collector.changed.set();time.sleep(.005)

    def test_missing_unreceived_request_is_cancelled_only_after_later_diagnostic(self):
        self.collector.register(self.epoch,self.request,100)
        self.wait(lambda:self.collector.last_error is not None)
        self.assertEqual(self.collector.pending(),1)
        self.collector.confirm_inference_barrier(self.barrier())
        self.wait(lambda:self.collector.pending()==0)
        entry=self.collector.entries[0]
        self.assertEqual(entry['state'],'cancelled')
        self.assertFalse(entry['prediction_received'])
        self.assertFalse(entry['submitted_to_ffw'])
        self.assertEqual(entry['cancellation_proof'],'later_diagnostic_completed')
        self.assertIsNone(self.collector.last_error)

    def test_missing_received_prediction_stays_an_error_even_with_diagnostic(self):
        self.collector.prediction({'epoch':self.epoch,'request_id':self.request,'source_time_ns':100})
        self.collector.confirm_inference_barrier(self.barrier())
        self.wait(lambda:self.collector.last_error is not None)
        self.assertEqual(self.collector.pending(),1)
        self.assertNotEqual(self.collector.entries[0]['state'],'cancelled')

    def test_generated_interrupted_prediction_is_collected_instead_of_discarded(self):
        self.client.available=True
        self.collector.register(self.epoch,self.request,100)
        self.collector.confirm_inference_barrier(self.barrier())
        self.wait(lambda:self.collector.pending()==0)
        self.assertEqual(self.collector.entries[0]['state'],'complete')

    def test_request_proven_not_sent_is_terminal_without_waiting_for_a_video(self):
        self.collector.register(self.epoch,self.request,100)
        self.collector.request_not_generated(self.request,'Cancelled before upload','client_not_submitted')
        self.assertEqual(self.collector.pending(),0)
        self.assertEqual(self.collector.entries[0]['state'],'cancelled')

    def test_barrier_from_another_server_is_rejected(self):
        result=self.barrier();result['server_url']='http://127.0.0.1:18009'
        with self.assertRaises(ValueError):self.collector.confirm_inference_barrier(result)

    def test_resumed_collector_uses_saved_diagnostic_without_new_inference(self):
        self.collector.register(self.epoch,self.request,100)
        self.close()
        folder=self.root/'diagnostics/pause';folder.mkdir(parents=True)
        (folder/'diagnostic.json').write_text(json.dumps(self.barrier()))
        self.collector=RunRollouts(self.root,self.client,resume=True)
        self.wait(lambda:self.collector.pending()==0)
        self.assertEqual(self.collector.entries[0]['state'],'cancelled')

    def test_pre_request_cancellation_is_explicit_and_sends_no_http_request(self):
        client=InferenceClient(self.client.url)
        client.cancel()
        connection=mock.Mock()
        with mock.patch('bg2_bridge.rpc.http.client.HTTPConnection',return_value=connection):
            with self.assertRaises(InferenceNotSubmitted):client._call('POST','/predict')
        connection.request.assert_not_called()


class QuietStatusTests(unittest.TestCase):
    def test_quiet_status_keeps_full_journal_without_flooding_terminal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);config=root/'profile.json';config.write_text(json.dumps(profile()))
            transport=mock.Mock();transport.diagnostics.return_value={'raw_wh':[672,376]}
            with mock.patch.object(gateway,'GatewayTransport',return_value=transport), \
                    mock.patch.object(gateway,'InferenceClient',side_effect=AssertionError('No model needed')), \
                    mock.patch.object(gateway.signal,'signal'),redirect_stdout(StringIO()) as output:
                gateway.main(['record-cameras','--profile',str(config),'--quiet-status',
                              '--duration','.025','--output',str(root/'recording')])
            self.assertNotIn('raw_wh',output.getvalue())
            self.assertIn('raw_wh',(root/'recording/events.jsonl').read_text())


if __name__=='__main__':unittest.main()
