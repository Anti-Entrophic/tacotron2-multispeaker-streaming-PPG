# tacotron2-streaming-PPG

training  

original:  input: &nbsp; text(sequence)-wav pairs
   
&nbsp;&nbsp;  [batch_size, max_len, 512]
   
now:  input: &nbsp; PPG-wav pairs

&nbsp;&nbsp;   [batch_size, frames, 音素个数]

如果对维度数据有问题，可以看一下原来的TextMelLoader。

The original TextMelLoader :

![image](https://github.com/Anti-Entrophic/tacotron2-multispeaker-streaming-PPG/blob/main/IMG/load_data.jpg)

目前已经可以开始训练了。speaker_embedding有点小bug，然后速度也比较慢。需要考虑。

inference:

![image](https://github.com/Anti-Entrophic/tacotron2-streaming-PPG/blob/main/IMG/infer.jpg)

进度： 

1、将原本的data_utils.py中的Text_Mel_Loader改成PPG_Mel_Loader，删去不必要的text_cleaners等内容

2、用Resemblyzer实现从wav中提取speaker_id的embedding（https://github.com/resemble-ai/Resemblyzer ），整合进PPG作为输入。（这步有点小bug）

3、测试tacotron2模型的训练（正在矩池云上跑）

4、改变infer部分的代码，主要是model.py中，Tacotron类的inference（要想想怎么送入decoder得到结果），以及Decoder类的inference（以片段的形式输出）

还需要保留一下前后的padding

5、wavenet前还需要一个buffer

## update  
