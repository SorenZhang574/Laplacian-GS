
from random import randint
import torch

class RandomCamSampler:
    """
    Replicates your original logic:
      - maintain a remaining list (viewpoint_stack)
      - each step randomly pop one camera (without replacement)
      - when empty, refill from full cam list
    """
    def __init__(self, cam_list):
        self._all = list(cam_list)
        self._stack = []
        self._idx = []
        self._refill()

    def _refill(self):
        self._stack = self._all.copy()
        self._idx = list(range(len(self._stack)))

    def next(self):
        if not self._stack:
            self._refill()
        ridx = randint(0, len(self._idx) - 1)
        cam = self._stack.pop(ridx)
        _ = self._idx.pop(ridx)
        return cam


class DictPrefetcher:
    """
    Prefetch tensors from CPU dicts to GPU with a lookahead queue.
    Schedules copy on a dedicated CUDA stream, returns tensors after wait_event.

    It only pins & copies the few prefetched items (won't pin the whole dict).
    """
    def __init__(self, device="cuda", lookahead=2, pin_cpu=True):
        assert lookahead >= 1
        self.device = torch.device(device)
        self.lookahead = int(lookahead)
        self.pin_cpu = pin_cpu
        self.stream = torch.cuda.Stream(device=self.device)
        self.buf = {}  # uid -> {"gt": gpu_tensor, "pred": gpu_tensor_or_None, "event": event}

    def _h2d_async(self, x):
        if x is None:
            return None
        if x.is_cuda:
            return x if x.device == self.device else x.to(self.device, non_blocking=True)
        if self.pin_cpu and (not x.is_pinned()):
            x = x.pin_memory()
        return x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def schedule(self, uid, gt_image_dict, pred_accumulated_dict=None, need_pred=False):
        if uid in self.buf:
            self.buf[uid]["ref"] += 1
            return

        gt_x = gt_image_dict[uid]
        pred_x = None
        if need_pred and (pred_accumulated_dict is not None):
            pred_x = pred_accumulated_dict[uid]

        evt = torch.cuda.Event()
        with torch.cuda.stream(self.stream):
            gt_gpu = self._h2d_async(gt_x)
            pred_gpu = self._h2d_async(pred_x) if (pred_x is not None) else None
            evt.record(self.stream)

        self.buf[uid] = {"gt": gt_gpu, "pred": pred_gpu, "event": evt, "ref": 1}

    def fetch(self, uid):
        item = self.buf[uid]
        torch.cuda.current_stream(device=self.device).wait_event(item["event"])

        item["ref"] -= 1
        gt, pred = item["gt"], item["pred"]

        if item["ref"] <= 0:
            del self.buf[uid]

        return gt, pred
