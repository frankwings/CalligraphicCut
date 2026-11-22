"""Draw the medial axes of cuts at selected steps over the then-current image."""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params  # noqa: E402
from calligraphic_cut.core import best_cut, remove_cut, gradients  # noqa: E402
from calligraphic_cut.io import load  # noqa: E402


def draw(img, xs, color=(255, 0, 0)):
    im = Image.fromarray(np.clip(np.rint(img * 255), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    d.line([(float(x), float(k)) for k, x in enumerate(xs)], fill=color, width=1)
    return im


def main(path, steps, out, **kw):
    p = Params(**kw)
    cur = load(path)
    panels = []
    for s in range(max(steps) + 1):
        e, xs = best_cut(cur, p)
        if s in steps:
            panels.append(draw(cur, xs))
            print(s, "E=%.3f" % e, "x range %.1f..%.1f" % (xs.min(), xs.max()))
        cur = remove_cut(cur, xs)
    W = sum(i.width for i in panels) + 10 * len(panels)
    c = Image.new("RGB", (W, panels[0].height), "white")
    x = 0
    for i in panels:
        c.paste(i, (x, 0))
        x += i.width + 10
    c.save(out)


if __name__ == "__main__":
    import json
    main(sys.argv[1], json.loads(sys.argv[2]), sys.argv[3], **json.loads(sys.argv[4] if len(sys.argv) > 4 else "{}"))
