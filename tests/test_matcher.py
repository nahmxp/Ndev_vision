import numpy as np
from PIL import Image
from core.vision_matcher import match_template
def test_finds_known_coordinate():
    canvas=np.zeros((100,120),dtype=np.uint8); ref=np.arange(300,dtype=np.uint8).reshape(20,15)
    canvas[31:51,47:62]=ref
    found=match_template(Image.fromarray(canvas),Image.fromarray(ref),min_scale=1,max_scale=1,scale_steps=1,stride=1)
    assert (found.x,found.y,found.width,found.height)==(47,31,15,20)
    assert found.score==1.0
