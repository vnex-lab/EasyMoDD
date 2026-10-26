"""Console system."""
class Console:
    def __init__(self,prompt="> "): self.prompt=prompt; self.history=[]
    def read(self): 
        try: line=input(self.prompt)
        except EOFError: return None
        self.history.append(line); return line
    def print(self,*a): print(*a)
    def run_loop(self,handler):
        while True:
            line=self.read()
            if line is None: break
            if line.strip() in ("exit","quit"): break
            print(handler(line))
