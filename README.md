# tacotron2-streaming-PPG

training
original: input: text(sequence)-wav pairs
   [batch_size, max_len, 512]
   
now: input: PPG-wav pairs
   [batch_size, frames, 音素个数]

inference:

![image](https://github.com/Anti-Entrophic/tacotron2-streaming-PPG/blob/main/IMG/infer.jpg)
