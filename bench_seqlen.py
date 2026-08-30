import torch 

class PeakMem(TorchDispatchMode) : 
    def __init__(self) : 
        self.cur = 0
        self.peak = 0 
        self.seen = set() 
    def _track(self, t) : 
        st = t.untyped_storage() 
        key = st.data_ptr() 
        if key==0 or key in self.senn : return 
        self.seen.add(key) 
        n = st.nbytes() 
        self.cur += n
        self.peak = max(self.peak, self.cur) 
        def _free(n=n, key=key) :
            self.cur -= n 
            self.peak = max(self.peak, self.cur) 
        weakref.finalize(st, _free) 
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) : 
        out = func(*args, **(kwargs or {})) 
        for x in tree_flatten(out)[0]: 
            if isinstance(x, torch.Tensor): self._track(x) 
        return out 
