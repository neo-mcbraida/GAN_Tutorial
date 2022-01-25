import tensorflow as tf

print(tf. __version__)


sys_details = tf.sysconfig.get_build_info()
cuda_version = sys_details["cuda_version"]
print(cuda_version)

cudnn_version = sys_details["cudnn_version"]  
print(cudnn_version)

print(tf.test.is_built_with_cuda())

print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))