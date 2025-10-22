import os
import numpy as np
import pywt
import wfdb


def wt(data, wavelet, level, threshold_factor=0.1):
    # 确保输入数据是一维数组
    Data = data.flatten() if data.ndim > 1 else data
    coeffs = pywt.wavedec(data=Data, wavelet=wavelet, level=level, mode='symmetric')
    cA, *cD = coeffs  # 分离近似系数和细节系数

    # 计算阈值
    threshold = threshold_factor * np.max(np.abs(cD[0]))

    # 对高频系数应用软阈值处理
    cD = [pywt.threshold(d, threshold, mode='soft') for d in cD]

    # 重构信号
    rdata = pywt.waverec([cA] + cD, wavelet=wavelet, mode='symmetric')
    return rdata


def process_mit_bih_dataset(data_path, wavelet='sym8', level=5, threshold_factor=0.1, save_path=None):
    # 获取数据集中的所有记录文件
    record_files = [f for f in os.listdir(data_path) if f.endswith('.dat')]

    for record_file in record_files:
        record_name = os.path.splitext(record_file)[0]
        print(f"Processing record: {record_name}")

        # 读取记录
        record = wfdb.rdrecord(os.path.join(data_path, record_name))

        # 创建保存路径
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            save_dir = save_path
        else:
            save_dir = data_path

        # 遍历每个信号通道
        denoised_signals = []
        for channel in range(record.p_signal.shape[1]):
            signal = record.p_signal[:, channel]
            denoised_signal = wt(signal, wavelet=wavelet, level=level, threshold_factor=threshold_factor)

            # 检查并限制信号值
            denoised_signal = np.clip(denoised_signal, -2147483648, 2147483647)
            denoised_signals.append(denoised_signal)

        # 将去噪后的信号保存为新的记录文件
        denoised_signals = np.array(denoised_signals).T
        wfdb.wrsamp(record_name=record_name, fs=record.fs, units=record.units, sig_name=record.sig_name,
                    p_signal=denoised_signals, write_dir=save_dir)


# 示例：处理 MIT-BIH 数据集并保存去噪后的数据
data_path = "D:/python/python项目/医学图像/mit-bih-arrhythmia-database-1.0.0"
save_path = "D:/python/python项目/医学图像/mit-bih-arrhythmia-database-denoised"  # 保存路径
process_mit_bih_dataset(data_path, wavelet='sym8', level=5, threshold_factor=0.1, save_path=save_path)
