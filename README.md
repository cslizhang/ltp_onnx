# ltp_onnx

#### 介绍
哈工大ltp，在自然语言处理领域,是一个功能比较全面的框架。但在实际应时，由于其改版后采用Torch训练，模型部署不太方便。该模块主要实现了ltp模型到onnx格式的转换调用，方便其他用户部署使用。

#### 使用说明

1. ***base_model.onnx***文件太大，请下载后放置model文件夹下。
链接: https://pan.baidu.com/s/1NcMc8LdAja1vxVc2v7VQ1Q  密码: 9doc
2. 句子转换成token id,已经有很多实现，这里直接调用的ltp，因此需要安装ltp。如有需要请自己写个转换，也很简单。

#### 致谢

感谢哈工大ltp工作组的各位大神！