# exo_rs Python shim; self-contained ABI stub for c-geek zenoh migration.
import asyncio

class PidfileError(Exception):
    pass

class Pidfile:
    def __init__(self, path, mode=0o644):
        self.path = path
        self.mode = mode

    def write(self):
        import os
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, self.mode)
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)

    def close(self):
        pass

    def as_raw_fd(self):
        return 0


class Connection:
    peer: str = ""

class Message:
    topic = ""
    data = b""

class _FromSwarm:
    Connection = Connection
    Message = Message

PyFromSwarm = _FromSwarm()
FromSwarm = _FromSwarm()

class Keypair:
    def __init__(self, *a, **kw):
        pass

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

    async def gossipsub_publish(self, topic, payload):
        return None

    async def gossipsub_subscribe(self, topic):
        return None

    async def gossipsub_unsubscribe(self, topic):
        return None

    async def recv(self):
        await asyncio.sleep(3600 * 24)
        return None

    def close(self):
        pass

