import tensorflow as tf

# 示例数据
B = 5  # 批次大小
a = tf.random.uniform((B, B))  # 假设的形状为 (B, B) 的矩阵 a

# 将矩阵 a 扩展成形状为 (B, B*B) 的矩阵 b，按顺序重复每个元素 B 次
b = tf.repeat(a, repeats=B, axis=1)

# 在 TensorFlow 1.x 中，使用会话来运行并输出张量的内容
with tf.Session() as sess:
    a_val, b_val = sess.run([a, b])
    print("Original matrix a:")
    print(a_val)
    print("Expanded matrix b:")
    print(b_val)
