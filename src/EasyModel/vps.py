"""Small VPS system (simulated)."""
class VPS:
    def __init__(self,name,cpu=1,ram=512): self.name=name; self.cpu=cpu; self.ram=ram; self.status="stopped"
    def start(self): self.status="running"
    def stop(self): self.status="stopped"
    def info(self): return {"name":self.name,"cpu":self.cpu,"ram":self.ram,"status":self.status}
class VPSManager:
    def __init__(self): self.vps={}
    def create(self,name,**kw): v=VPS(name,**kw); self.vps[name]=v; return v
    def list(self): return [v.info() for v in self.vps.values()]
