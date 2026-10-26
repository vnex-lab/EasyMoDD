"""Encryption helpers (stdlib)."""
import base64, hashlib, os

def _key(secret): return hashlib.sha256(secret.encode()).digest()

def encrypt(text, secret):
    k=_key(secret); data=text.encode()
    out=bytes(b ^ k[i % len(k)] for i,b in enumerate(data))
    return base64.b64encode(out).decode()

def decrypt(token, secret):
    k=_key(secret); data=base64.b64decode(token)
    return bytes(b ^ k[i % len(k)] for i,b in enumerate(data)).decode()

def hash_password(pw, salt=None):
    salt=salt or os.urandom(16).hex()
    h=hashlib.sha256((salt+pw).encode()).hexdigest()
    return salt+":"+h

def verify_password(pw, stored):
    salt, h = stored.split(":",1)
    return hashlib.sha256((salt+pw).encode()).hexdigest()==h
