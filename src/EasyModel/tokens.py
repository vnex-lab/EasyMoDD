"""Token system."""
import secrets, time, hashlib
class TokenManager:
    def __init__(self): self.tokens={}
    def issue(self,subject,ttl=3600): t=secrets.token_urlsafe(48); self.tokens[t]={"subject":subject,"expires":time.time()+ttl}; return t
    def verify(self,t):
        e=self.tokens.get(t)
        if not e: return False
        if time.time()>e["expires"]: del self.tokens[t]; return False
        return e["subject"]
    def revoke(self,t): self.tokens.pop(t,None)
