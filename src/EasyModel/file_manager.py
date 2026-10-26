"""File management utilities (stdlib only)."""
import os, shutil, hashlib, json
from pathlib import Path

class FileManager:
    def __init__(self, root="."):
        self.root=Path(root).resolve()
    def list(self, sub=""):
        p=self.root/sub
        return [str(x.relative_to(self.root)) for x in p.iterdir()] if p.exists() else []
    def read(self, path):
        return (self.root/path).read_text(encoding="utf-8",errors="replace")
    def write(self, path, content):
        p=self.root/path; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding="utf-8"); return str(p)
    def delete(self, path):
        p=self.root/path
        if p.is_dir(): shutil.rmtree(p)
        elif p.exists(): p.unlink()
    def copy(self, src, dst):
        s=self.root/src; d=self.root/dst; d.parent.mkdir(parents=True,exist_ok=True)
        if s.is_dir(): shutil.copytree(s,d)
        else: shutil.copy2(s,d)
    def move(self, src, dst): self.copy(src,dst); self.delete(src)
    def hash(self, path, algo="sha256"):
        h=hashlib.new(algo)
        with open(self.root/path,"rb") as f:
            for c in iter(lambda:f.read(65536),b""): h.update(c)
        return h.hexdigest()
    def search(self, pattern, sub=""):
        import fnmatch
        out=[]
        for p in (self.root/sub).rglob("*"):
            if p.is_file() and fnmatch.fnmatch(p.name,pattern): out.append(str(p.relative_to(self.root)))
        return out
