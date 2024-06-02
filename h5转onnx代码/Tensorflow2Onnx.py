# 读取h5模型转换为onnx模型，教程来自于https://github.com/onnx/tensorflow-onnx/blob/master/tutorials/keras-resnet50.ipynb
import tensorflow as tf
import numpy as np
import cv2
import tf2onnx
import onnxruntime as rt
 
######################################################################################################################
# 图片预处理，对输入图片进行等比例拉伸至指定尺寸，不足的地方填0，同时把像素值归一化到[-1, 1]
def image_preprocess(image, target_length, value=0.0, method=0):
    image = image.astype("float32")
    h, w, _ = image.shape                               # 获得原始尺寸
    ih, iw  = target_length, target_length              # 获得目标尺寸
    scale = min(iw/w, ih/h)                             # 实际拉伸比例
    nw, nh  = int(scale * w), int(scale * h)            # 实际拉伸后的尺寸
    image_resized = cv2.resize(image, (nw, nh))         # 实际拉伸图片
    image_paded = np.full(shape=[ih, iw, 3], fill_value=value)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized  # 居中填充图片
    if method == 0:
        image_paded = image_paded / 255.                # 图片归一化
    elif method == 1:
        image_paded = image_paded / 127.5 - 1.0         # 图片标准化
    return image_paded
 
# 读取图片并预处理
image_path = "C:/Users/Ruiling/Desktop/click-samples/0.jpg"
# "C:/Users/Ruiling/Desktop/click-samples/90.jpg"  # 图片路径
image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype("float32")  # 读取图片
image = image_preprocess(image, 224, 0, 1)                                         # 图片预处理
image = np.expand_dims(image, axis=0).astype(np.float32)                           # 图片维度扩展
 
######################################################################################################################
# 读取h5模型
model = tf.keras.models.load_model("converted_keras\keras_model.h5")
 
# 推理h5模型
preds = model.predict(image)
 
# 保存h5模型为tf的save_model格式
# model.save("./" + model.name))
 
######################################################################################################################
# 定义模型转onnx的参数
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)  # 输入签名参数，(None, 128, 128, 3)决定输入的size
output_path = model.name + ".onnx"                                   # 输出路径
 
# 转换并保存onnx模型，opset决定选用的算子集合
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)  # 查看输出名称，后面推理用的到
 
 
######################################################################################################################
# 读取onnx模型，安装GPUonnx，并设置providers = ['GPUExecutionProvider']，可以实现GPU运行onnx
# providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path)
 
# 推理onnx模型
onnx_pred = m.run(output_names, {"input": image})
 
# 对比两种模型的推理结果
print('Keras Predicted:', preds)
print('ONNX Predicted:', onnx_pred[0])
 
# make sure ONNX and keras have the same results
np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-4)