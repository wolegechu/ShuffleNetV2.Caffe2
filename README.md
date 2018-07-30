# ShuffleNetV2.Caffe2

## TODO

* [ ] Train/Test mode
* [ ] Detection: backbone for Faster R-CNN
* [ ] Detection: backbone for FPN
* [ ] Pretrain model

```python
from caffe2.python import cnn
from shufflenetv2 import add_ShuffleNet_V2

cnn_model = cnn.CNNModelHelper()

add_ShuffleNet_V2(cnn_model)
workspace.RunNetOnce(cnn_model.param_init_net)

workspace.FeedBlob("data", np.random.randn(8, 3, 224, 224).astype(np.float32))
workspace.RunNetOnce(cnn_model.Proto())

print(workspace.FetchBlob('conv_5'))
```
