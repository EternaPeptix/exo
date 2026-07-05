"""Tests for the configurable ring link priority (EXO_RING_LINK_PRIORITY)."""
import pytest

from exo.master.placement_utils import _ring_link_priority, find_ip_prioritised
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)


def _sock(ip: str) -> SocketConnection:
    return SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/1234"))


def _net(ip_types: dict[str, str]) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name=f"if{i}", ip_address=ip, interface_type=t)
            for i, (ip, t) in enumerate(ip_types.items())
        ]
    )


def test_ring_priority_default_prefers_ethernet(monkeypatch):
    monkeypatch.delenv("EXO_RING_LINK_PRIORITY", raising=False)
    p = _ring_link_priority()
    assert p["ethernet"] < p["thunderbolt"], "default ring priority: ethernet before thunderbolt"


def test_ring_priority_override_prefers_thunderbolt(monkeypatch):
    """EXO_RING_LINK_PRIORITY=thunderbolt,... makes TB win for Mac<->Mac over 25G ethernet."""
    monkeypatch.setenv("EXO_RING_LINK_PRIORITY", "thunderbolt,ethernet,wifi,unknown")
    p = _ring_link_priority()
    assert p["thunderbolt"] < p["ethernet"], "override: thunderbolt before ethernet"


def test_find_ip_prioritised_default_picks_ethernet(monkeypatch):
    """Two Macs with both 25G ethernet + TB5 socket: default picks ethernet."""
    monkeypatch.delenv("EXO_RING_LINK_PRIORITY", raising=False)
    a, b = NodeId("macA"), NodeId("macB")
    topo = Topology()
    topo.add_node(a); topo.add_node(b)
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("10.0.0.2")))
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("169.254.10.2")))
    other_net = _net({"10.0.0.2": "ethernet", "169.254.10.2": "thunderbolt"})
    chosen = find_ip_prioritised(a, b, topo, {b: other_net}, ring=True)
    assert chosen == "10.0.0.2", f"default should pick ethernet, got {chosen}"


def test_find_ip_prioritised_override_picks_thunderbolt(monkeypatch):
    """With EXO_RING_LINK_PRIORITY=thunderbolt,..., Mac<->Mac picks the TB5 IP."""
    monkeypatch.setenv("EXO_RING_LINK_PRIORITY", "thunderbolt,ethernet,wifi,unknown")
    a, b = NodeId("macA"), NodeId("macB")
    topo = Topology()
    topo.add_node(a); topo.add_node(b)
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("10.0.0.2")))
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("169.254.10.2")))
    other_net = _net({"10.0.0.2": "ethernet", "169.254.10.2": "thunderbolt"})
    chosen = find_ip_prioritised(a, b, topo, {b: other_net}, ring=True)
    assert chosen == "169.254.10.2", f"override should pick thunderbolt, got {chosen}"


def test_jaccl_coordinator_path_still_prefers_thunderbolt(monkeypatch):
    """Non-ring (JACCL) path is not affected by EXO_RING_LINK_PRIORITY — TB always first."""
    monkeypatch.setenv("EXO_RING_LINK_PRIORITY", "ethernet,thunderbolt")
    a, b = NodeId("macA"), NodeId("macB")
    topo = Topology()
    topo.add_node(a); topo.add_node(b)
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("10.0.0.2")))
    topo.add_connection(Connection(source=a, sink=b, edge=_sock("169.254.10.2")))
    other_net = _net({"10.0.0.2": "ethernet", "169.254.10.2": "thunderbolt"})
    chosen = find_ip_prioritised(a, b, topo, {b: other_net}, ring=False)
    assert chosen == "169.254.10.2", f"JACCL path must prefer TB regardless of env, got {chosen}"
