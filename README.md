# tacotron2-streaming-PPG

training  

original:  input: &nbsp; text(sequence)-wav pairs
   
&nbsp;&nbsp;  [batch_size, max_len, 512]
   
now:  input: &nbsp; PPG-wav pairs

&nbsp;&nbsp;   [batch_size, frames, 音素个数]

inference:

![image](https://github.com/Anti-Entrophic/tacotron2-streaming-PPG/blob/main/IMG/infer.jpg)

进度： 

1、将原本的data_utils.py中的Text_Mel_Loader改成PPG_Mel_Loader，删去不必要的text_cleaners等内容

2、用Resemblyzer实现从wav中提取speaker_id的embedding（https://github.com/resemble-ai/Resemblyzer ），整合进PPG作为输入

3、测试tacotron2模型的训练

4、改变infer部分的代码，主要是model.py中，Tacotron类的inference（要想想怎么送入decoder得到结果），以及Decoder类的inference（以片段的形式输出）

还需要保留一下前后的padding

5、wavenet前还需要一个buffer

## update  
2023.2.12 开始干活
