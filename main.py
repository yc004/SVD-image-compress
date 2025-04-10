import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# 对图像进行SVD压缩
def svd_compression(image, k):
    # 打印压缩开始信息
    print(f"正在对图像进行 SVD 压缩，k 值为 {k}...")

    # 执行奇异值分解，得到三个矩阵
    # U: 左奇异矩阵，s: 奇异值向量，Vt: 右奇异矩阵的转置
    U, s, Vt = np.linalg.svd(image)

    # 截取前k个奇异向量（左奇异矩阵的前k列）
    Uk = U[:, :k]

    # 构建奇异值对角矩阵（仅取前k个奇异值）
    sk = np.diag(s[:k])

    # 截取右奇异矩阵的前k行（转置后的前k列）
    Vtk = Vt[:k, :]

    # 打印压缩完成信息
    print(f"图像 SVD 压缩完成，k 值为 {k}。")

    # 返回压缩后的图像矩阵：Uk * sk * Vtk
    return np.dot(np.dot(Uk, sk), Vtk)


# 照片路径
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg"]
# 读取照片
print("开始读取照片...")
images = []
for path in image_paths:
    img = cv2.imread(path, 0)
    if img is not None:
        print(f"成功读取照片: {path}")
        images.append(img)
    else:
        print(f"无法读取照片: {path}")
print("照片读取完成。")

# 定义不同的k值
ks = [10, 50, 100]

# 创建结果文件夹
result_folder = "result"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 显示结果
plt.figure(figsize=(15, 10))
for i, img in enumerate(images):
    plt.subplot(5, len(ks) + 1, i * (len(ks) + 1) + 1)
    plt.imshow(img, cmap="gray")
    if i == 0:
        plt.title("Original Image")
    for j, k in enumerate(ks):
        compressed_img = svd_compression(img, k)
        # 保存压缩后的图像到结果文件夹
        compressed_img_path = os.path.join(
            result_folder, f"compressed_image_{i + 1}_k_{k}.png"
        )
        plt.imsave(compressed_img_path, compressed_img, cmap="gray")
        print(f"压缩图像 {compressed_img_path} 已保存，k 值为 {k}。")

        plt.subplot(5, len(ks) + 1, i * (len(ks) + 1) + j + 2)
        plt.imshow(compressed_img, cmap="gray")
        if i == 0:
            plt.title(f"k = {k}")

plt.tight_layout()
plt.show()
