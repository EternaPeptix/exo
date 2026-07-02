# exo_rs Python stub ABI for zenoh migration (c-geek/main compatible).
import os
import asyncio

class PidfileError(Exception):
    pass

class Pidfile:
    def __init__(self, path, mode=0o644):
        self.path = path
        self.mode = mode
        self._fd = None

    def write(self):
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, self.mode)
        try:
            os.ftruncate(self._fd, 0)
            os.write(self._fd, str(os.getpid()).encode())
        finally:
            os.close(self._fd)
            self._fd = None

    def close(self):
        pass

    def as_raw_fd(self):
        return self._fd if self._fd is not None else 0


class _Connection:
    pass


class _Message:
    def __init__(self, topic, data):
        self.topic = topic
        self.data = data


class _FromSwarm:
    Connection = _Connection
    Message = _Message


PyFromSwarm = _FromSwarm()
FromSwarm = _FromSwarm()


class _Keypair:
    def __init__(self, *a, **kw):
        pass


Keypair = _Keypair()


class NetworkingHandle:
    @classmethod
    def new(cls, identity, namespace, listen_port, discovery_service_port):
        obj = cls(identity, [], listen_port)
        obj.discovery_service_port = discovery_service_port
        return obj

    def __init__(self, identity, bootstrap_peers, listen_port, discovery_service_port=None):
        self.identity = identity
        self.bootstrap_peers = list(bootstrap_peers or [])
        self.listen_port = listen_port
        self.discovery_service_port = discovery_service_port
        self._subscribers = set()

    def gossipsub_publish(self, topic, payload):
        self._subscribers.add(topic)

    async def gossipsub_subscribe(self, topic):
        self._subscribers.add(topic)

    async def gossipsub_unsubscribe(self, topic):
        self._subscribers.discard(topic)

    async def recv(self):
        # Wait indefinitely; real implementation would use zenoh session.
        await asyncio.sleep(3600 * 24)
        return None

    def close(self):
        self._subscribers.clear()

