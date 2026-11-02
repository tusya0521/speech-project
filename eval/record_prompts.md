# What to record

Record each line **in your own voice** and save it with the exact filename shown in `eval/audio/`.
- Format: WAV, mono, 16 kHz if your recorder allows it (any WAV/MP3 works; it's resampled).
- Use a quiet room and your normal laptop microphone (the realistic setting). Speak naturally and don't over-enunciate.
- Read the sentence exactly as written. If you stumble, re-record.
- Free recorder: Audacity (set Project Rate to 16000 Hz, then File > Export > Export as WAV), or the Windows Voice Recorder (then convert to WAV).

## A. WER sentences (25 clips): `eval/audio/wer_XX.wav`

The reference text lives in `eval/wer_set.csv`. Sentences 1 to 12 are tagged *plain*, 13 to 25 *technical*.

| File | Say exactly |
|---|---|
| wer_01.wav | Can you tell me what this chapter is about |
| wer_02.wav | Please read the next page out loud |
| wer_03.wav | I would like a shorter answer this time |
| wer_04.wav | What did the teacher say about the exam |
| wer_05.wav | Summarise pages three to five for me |
| wer_06.wav | Could you speak a little more slowly |
| wer_07.wav | Which page talks about the main idea |
| wer_08.wav | I did not understand the last part |
| wer_09.wav | Give me an example from my notes |
| wer_10.wav | How long should my revision take tonight |
| wer_11.wav | Repeat the answer one more time please |
| wer_12.wav | What is the difference between these two ideas |
| wer_13.wav | How are mel frequency cepstral coefficients computed |
| wer_14.wav | What does the Viterbi algorithm find in a hidden Markov model |
| wer_15.wav | Explain connectionist temporal classification and the blank symbol |
| wer_16.wav | Why does Kneser Ney smoothing help an n gram language model |
| wer_17.wav | What is the role of the discrete cosine transform |
| wer_18.wav | How does beam search prune unlikely hypotheses |
| wer_19.wav | What is a weighted finite state transducer |
| wer_20.wav | Describe the conformer encoder architecture |
| wer_21.wav | How does wav2vec learn from unlabelled audio |
| wer_22.wav | What does the Baum Welch algorithm estimate |
| wer_23.wav | How is perplexity related to the language model |
| wer_24.wav | What is the recurrent neural network transducer |
| wer_25.wav | Why are triphones tied using decision trees |

## B. QA questions (25 clips): `eval/audio/qa_XX.wav`

These are the questions in `eval/qa_set.csv`. 21 to 25 are deliberately off-topic, and the app should refuse them.

| File | Say exactly |
|---|---|
| qa_01.wav | What are the four parts of a classical speech recognition system? |
| qa_02.wav | Why is automatic speech recognition difficult? |
| qa_03.wav | What is the fundamental frequency of the voice? |
| qa_04.wav | What sampling rate does Whisper use? |
| qa_05.wav | Why is speech analysed in short frames? |
| qa_06.wav | What is the mel scale? |
| qa_07.wav | How are MFCCs computed? |
| qa_08.wav | How many dimensions does an MFCC vector with deltas have? |
| qa_09.wav | How many states does a phoneme HMM usually have? |
| qa_10.wav | What does the Viterbi algorithm do? |
| qa_11.wav | What is a triphone? |
| qa_12.wav | What is Kneser-Ney smoothing used for? |
| qa_13.wav | How is perplexity defined? |
| qa_14.wav | What is beam width in decoding? |
| qa_15.wav | What is a lattice? |
| qa_16.wav | What is the blank symbol in CTC? |
| qa_17.wav | Why is RNN-T used for streaming recognition? |
| qa_18.wav | How much training data was Whisper trained on? |
| qa_19.wav | What is vocabulary biasing? |
| qa_20.wav | How is word error rate calculated? |
| qa_21.wav | Who won the football world cup in 2018? |
| qa_22.wav | What is the capital city of Australia? |
| qa_23.wav | How do I bake a chocolate cake? |
| qa_24.wav | What is the boiling point of water at sea level? |
| qa_25.wav | Who wrote the novel Pride and Prejudice? |

When all 50 files exist, run `scripts/verify.ps1` (or the eval commands in the README).
