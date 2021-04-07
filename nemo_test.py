import numba
numba.__version__
import nemo
import nemo.collections.asr as nemo_asr
import gc
import torch


asr_model = nemo_asr.models.EncDecCTCModel.restore_from("QuartzNet15x5Base-Zh.nemo")
wave_file1 = ["/home/nvidia/0_NVDC_Visual/test/t1.wav"]
result = asr_model.transcribe(paths2audio_files = wave_file1,batch_size=1)
print(result)