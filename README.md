# Blood Pressure Estimation from Photoplethysmography by Transformer Networks

The continuous and non-invasive estimation of blood pressure (BP) is significant in monitoring
for early detection and management of cardiovascular disease. In this work, a novel approach
using an Informer Neural Network has been implemented inspired by Ma et al. 

The approach is based on the continuous BP waveform estimation using single photoplethysmogram
(PPG) signals. The Informer processes 8-s signal segments into an encoder-decoder
structure to capture time-series dependencies and patterns by using generative pre-trained
transformer (GPT) based attention mechanism, the ProbSparse Multi-Head Self-Attention
(MHSA) respectively. 

The proposed approach demonstrates a high correlation between the
predicted and reference BP signals with a Pearson correlation coefficient of R = 0.92. However,
the estimation of BP values, in particular the diastolic blood pressure (DBP) and systolic
blood pressure (SBP) values, show deficient results with −10.99 ± 12.69 mmHg for DBP and
−41.01±22.30 mmHg for SBP (ME ± SDE), which has been found to be related with a fundamentally
missing learning process of the Informer during training. These results do not meet
the standards set by the Association for Advancement of Medical Instrumentation (AAMI).
