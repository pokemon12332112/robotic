"""GAM-only simulator protocol; works in the separate Isaac Python environment."""
from collections import deque
import time
import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

POLICY_TYPE = 'gam_object_understanding'
HORIZON = 8


def observation(joints, images, instruction):
    state = np.asarray(joints, dtype=np.float32)
    if state.shape != (9,) or not np.isfinite(state).all():
        raise ValueError('Expected nine finite joint positions')
    return {'state': state, 'top': images['top']['rgb'], 'wrist': images['wrist']['rgb'], 'prompt': instruction}


def action_chunk(value):
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.shape != (HORIZON, 9) or not np.isfinite(chunk).all():
        raise ValueError('Expected eight finite nine-joint absolute targets')
    return chunk


class LocalPolicy:
    def __init__(self, port, *, expected_checkpoint=None):
        self.socket = connect(f'ws://127.0.0.1:{port}', compression=None, max_size=8 * 1024**2)
        try:
            metadata = msgpack_numpy.unpackb(self.socket.recv(timeout=30))
            if metadata['policy_type'] != POLICY_TYPE or metadata['horizon'] != HORIZON:
                raise ValueError('Unexpected policy server')
            if expected_checkpoint is not None and metadata['checkpoint'] != expected_checkpoint:
                raise ValueError('Unexpected checkpoint')
        except BaseException:
            self.close()
            raise

    def reset(self, seed):
        # The model is deterministic and memoryless (H=1).
        pass

    def infer(self, obs):
        self.socket.send(msgpack_numpy.packb(obs))
        message = self.socket.recv(timeout=180)
        if isinstance(message, str):
            raise RuntimeError(message)
        return action_chunk(msgpack_numpy.unpackb(message)['actions'])

    def close(self):
        self.socket.close()


class RecedingHorizonController:
    def __init__(self, policy, execution_horizon=4):
        if not 1 <= execution_horizon <= HORIZON:
            raise ValueError('Execution horizon must be 1..8')
        self.policy, self.execution_horizon = policy, execution_horizon
        self.pending = deque()
        self.inference_times = []

    def next_action(self, joints, images, instruction):
        if not self.pending:
            start = time.monotonic()
            actions = action_chunk(self.policy.infer(observation(joints, images, instruction)))
            self.inference_times.append(time.monotonic() - start)
            self.pending.extend(actions[:self.execution_horizon].copy())
        return self.pending.popleft()
