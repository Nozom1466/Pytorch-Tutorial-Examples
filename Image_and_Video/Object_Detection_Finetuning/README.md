## Takeaways

> Code in the tutorial CANNOT get the same performance as the proposed result with scores below `0.5` on average. (Notice `torchvision` version difference)
> Image data type in  `Visualization` part is really a mess....😰:
> (QUESTION) Is there any difference between training on CPU and training on GPU ? (Better performance on Colab GPU for me...)


### Platform
Despite the complex environment and outdated preinstalled packages(cases), It's always a better choice to take advantage of the GPU resource than to run the code locally. The `XiaoXinPro 16IHU 2021` features a 2-GB Nvidia GPU, which is not enough to run a decent model, whereas `Colab` or `Kaggle` notebook offers 15-GB GPU with various types.


**DO NOT HESITATE TO LAUNCH A PROJECT ON COLAB! 😭**


### Styled Python coding from `Google`
[coding with `Google` style](https://google.github.io/styleguide/pyguide.html)


### Techniques of creating masks
create different masks for each pedestrian
```python
import numpy as np

mask = np.array([[0, 0, 1, 1],
                 [0, 2, 2, 0],
                 [3, 3, 0, 0]])

obj_ids = np.unique(mask)[1:]
print(obj_ids[:, None, None])
# [[[1]]
#  [[2]]
#  [[3]]]

masks = mask == obj_ids[:, None, None]  # broadcast in numpy
print(masks)
# [[[False False  True  True]
#   [False False False False]
#   [False False False False]]
#  [[False False False False]
#   [False  True  True False]
#   [False False False False]]
#  [[False False False False]
#   [False False False False]
#   [ True  True False False]]]
```


### ROI Pooling
- [Anchor generator](https://tanalib.com/faster-rcnn-anchor/)
- [ROI Pooling & ROI Align](https://zhuanlan.zhihu.com/p/73138740)
- [Faster RCNN](https://zhuanlan.zhihu.com/p/145842317)









