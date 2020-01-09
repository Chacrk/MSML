# MSML
code for paper

### 1. Environment
#### 1.1. Requirements
1. Python 3.X
2. PyTorch (ver>=0.4)
3. Numpy
4. mini-ImageNet dataset

#### 1.2. Testing Environment
##### 1.2.1 Software:
* Ubuntu 16.04
* Python 3.6.1
* PyTorch 1.0.1
* Numpy 1.17.2

##### 1.2.2 Hardware:
* CPU: Intel Xeon E5-2620 v4 @2.10GHz with 8 Cores
* GPU: NVIDIA TITAN Xp with CUDA 8.0.61

### 2. File Structure

MSML  
└─**data**  
&emsp;└─miniimagenet  
&emsp;&emsp;&emsp;├─images  
&emsp;&emsp;&emsp;&emsp;├─nxx.jpg  
&emsp;&emsp;&emsp;&emsp;├─...  
&emsp;&emsp;&emsp;├─train.csv  
&emsp;&emsp;&emsp;├─val.csv  
&emsp;&emsp;&emsp;├─test.csv  
&emsp;─proc_images.py  
└─**meta**  
&emsp;&emsp;├─main.py  
&emsp;&emsp;├─model.py  
&emsp;&emsp;├─net.py  
&emsp;&emsp;├─data_provider.py  
└─**pretrain**  
&emsp;&emsp;├─pretrain.py  
&emsp;&emsp;├─data_provider_pretrain.py  
&emsp;&emsp;├─net_pretrain.py  
### 3. Experiment Details
#### 3.1. Computing resource usage
|  | RAM | GPU Memory|
| --- | --- | --- |
| Pretrain Phase| 1500MB | 6267MB |
| 5-way 1-shot | 1800MB | 8767MB |
| 5-way 5-shot | 1800MB | 10157MB |

#### 3.2. Speed
|  | iter/s | total time |
| --- | --- | --- |
| Pretrain | 2.64 | 1h-23m-45s |
| 5-way 1-shot | 0.7 | 1h-23m-45s |
| 5-way 5-shot | 0.3 | 1h-23m-45s |
