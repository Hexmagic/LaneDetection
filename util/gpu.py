import sys
import time
from datetime import datetime

from termcolor import colored

import pynvml


def wait_gpu(need=4, sleep=5):
    '''
    param:
        need: 需要几G的空间，需要自己估算
        sleep: 检查间隔，秒为单位
    '''
    pynvml.nvmlInit()
    cnt = pynvml.nvmlDeviceGetCount()
    G = 1 << 30
    print(f"Computer Has {cnt} GPU")
    stext = colored("Waiting GPU...", color="yellow", attrs=["blink"])
    gpus = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(cnt)]
    while True:
        for i, gpu in enumerate(gpus):
            info = pynvml.nvmlDeviceGetMemoryInfo(gpu)
            free = info.free / G
            if free > need:
                print(f"Find GPU {i} Has Free Memory {free}G")
                return i
        sys.stdout.write('\r')
        sys.stdout.flush()
        date = datetime.now().strftime('%m-%d %H:%M:%S')
        stext = colored(f"{date}: Waiting GPU...",
                        on_color="on_magenta",
                        color="yellow",
                        attrs=["blink"])
        sys.stdout.write(stext)
        time.sleep(sleep)
