"""Container system (lightweight in-process)."""
class Container:
    def __init__(self,name,image="base"): self.name=name; self.image=image; self.env={}; self.running=False
    def start(self): self.running=True
    def stop(self): self.running=False
    def exec(self,cmd): return f"[{self.name}] {cmd}"
class ContainerManager:
    def __init__(self): self.containers={}
    def create(self,name,image="base"): c=Container(name,image); self.containers[name]=c; return c
    def list(self): return list(self.containers.values())
    def get(self,name): return self.containers.get(name)
