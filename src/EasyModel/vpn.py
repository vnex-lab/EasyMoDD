"""VPN system (simulated)."""
import secrets
class VPN:
    def __init__(self,name): self.name=name; self.key=secrets.token_hex(16); self.connected=False
    def connect(self): self.connected=True
    def disconnect(self): self.connected=False
class VPNManager:
    def __init__(self): self.vpns={}
    def create(self,name): v=VPN(name); self.vpns[name]=v; return v
    def list(self): return [(n,v.connected) for n,v in self.vpns.items()]
