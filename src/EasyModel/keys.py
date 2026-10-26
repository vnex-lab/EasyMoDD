"""Key system."""
import secrets, time
class KeyManager:
    def __init__(self): self.keys={}
    def generate(self,name,ttl=None): k=secrets.token_urlsafe(32); self.keys[name]={"value":k,"expires":time.time()+ttl if ttl else None}; return k
    def get(self,name):
        e=self.keys.get(name)
        if not e: return None
        if e["expires"] and time.time()>e["expires"]: del self.keys[name]; return None
        return e["value"]
    def revoke(self,name): self.keys.pop(name,None)
