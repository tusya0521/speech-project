"""Build the original 10-page sample study document used by the evaluation (one topic per page)."""
import os

import pymupdf

PAGES = [
("Introduction to Automatic Speech Recognition",
"""Automatic speech recognition, usually shortened to ASR, is the task of converting a spoken audio signal into written text. A recogniser receives a waveform and must output the most likely sequence of words that the speaker said.

ASR is difficult because speech is continuous. There are no clear pauses between words, the same word sounds different when spoken by different people, and background noise, microphones and room echo all change the signal.

A classical ASR system has four parts. The front end turns audio into acoustic features. The acoustic model scores how well those features match each speech sound. The pronunciation lexicon maps words to sequences of phonemes. The language model scores how likely a word sequence is in the language. A decoder searches for the word sequence with the best combined score.

Modern end-to-end systems replace several of these parts with a single neural network that maps features directly to characters or sub-word tokens. Even so, the classical decomposition is still the clearest way to understand what any recogniser must do.

ASR is used in voice assistants, dictation software, automatic captioning, call centre analytics and accessibility tools. For students with visual impairments or dyslexia, speech interfaces can remove the need to read or type."""),
("Speech Signals and Digital Audio",
"""Speech is produced when air from the lungs passes through the vocal folds and the vocal tract. Vibration of the vocal folds creates voiced sounds such as vowels, while turbulent air through narrow constrictions creates unvoiced sounds such as the s in sun.

The rate at which the vocal folds vibrate is the fundamental frequency, heard as pitch. Typical adult male voices are around 120 hertz and adult female voices around 210 hertz. The shape of the vocal tract creates resonances called formants, and the first two formants largely determine which vowel is heard.

To process speech on a computer, the analogue signal is sampled. The sampling rate is the number of samples taken per second. By the Nyquist theorem, the sampling rate must be at least twice the highest frequency we want to keep. Telephone speech uses 8 kilohertz sampling, while most modern ASR systems, including Whisper, use 16 kilohertz.

Each sample is quantised to a fixed number of bits. Sixteen-bit audio gives 65,536 possible amplitude levels, which is enough for speech.

Because speech changes over time, it is analysed in short frames, usually 25 milliseconds long with a 10 millisecond shift. Within one frame the signal is assumed to be roughly stationary. A window function such as the Hamming window is applied to each frame to reduce edge effects."""),
("Feature Extraction: Spectrograms and MFCCs",
"""Raw waveform samples are not a convenient input for classical recognisers, so each frame is converted into a compact feature vector.

The first step is the short-time Fourier transform, which gives the energy at each frequency within a frame. Plotting these energies over time gives a spectrogram, an image in which formants and harmonics are visible as dark bands.

Human hearing is more sensitive to differences between low frequencies than between high frequencies. The mel scale models this. A mel filterbank is a set of overlapping triangular filters spaced evenly on the mel scale. Summing the spectral energy inside each filter and taking the logarithm gives log mel filterbank energies.

Mel frequency cepstral coefficients, or MFCCs, are obtained by applying the discrete cosine transform to the log mel energies and keeping the first 12 or 13 coefficients. The discrete cosine transform decorrelates the features, which suited the diagonal Gaussian models used in older systems.

Delta and delta-delta coefficients, the first and second time derivatives of the MFCCs, are often appended to capture how the spectrum changes. Thirteen MFCCs with deltas and delta-deltas give a 39 dimensional feature vector.

Modern neural systems such as Whisper usually skip the cosine transform and use 80 log mel filterbank channels directly, because neural networks can learn from correlated inputs."""),
("Hidden Markov Models for Acoustic Modelling",
"""For several decades the hidden Markov model, or HMM, was the standard acoustic model. An HMM is a set of hidden states connected by transition probabilities. At each time step the model is in one state and emits an observation, here a feature vector, according to that state's emission distribution.

In speech, each phoneme is typically modelled by a left-to-right HMM with three emitting states, representing the beginning, middle and end of the sound. Word models are formed by joining phoneme models according to the pronunciation lexicon.

Emission probabilities were traditionally modelled with Gaussian mixture models, giving the GMM-HMM system. Later, deep neural networks replaced the Gaussian mixtures and predicted the probability of each state given a window of frames. These hybrid DNN-HMM systems gave large accuracy gains around 2012.

Three classic algorithms are used with HMMs. The forward algorithm computes the probability of an observation sequence. The Viterbi algorithm finds the single most likely state sequence and is used for decoding. The Baum-Welch algorithm, a form of expectation maximisation, trains the model parameters from data.

Context-dependent models called triphones model a phoneme together with its left and right neighbours, because the sound of a phoneme depends strongly on the sounds around it. Decision trees are used to tie similar triphone states so that there is enough training data for each."""),
("Language Models",
"""A language model assigns a probability to a sequence of words. In recognition it helps choose between acoustically similar hypotheses such as recognise speech and wreck a nice beach.

The classic approach is the n-gram model, which assumes that each word depends only on the previous n minus one words. A trigram model estimates the probability of a word given the two words before it, using counts from a large text corpus.

Many valid word sequences never appear in the training text, so raw counts give them zero probability. Smoothing methods fix this by moving some probability mass to unseen events. Kneser-Ney smoothing is the most widely used and performs well in practice.

Perplexity measures how well a language model predicts held-out text. It is the exponential of the average negative log probability per word, and lower perplexity means a better model.

Neural language models based on recurrent networks or transformers predict the next word using a much longer context and usually achieve far lower perplexity than n-gram models. In ASR they are often used to rescore a list of the best hypotheses produced by a first decoding pass.

The relative weight of the acoustic model and the language model is controlled by a language model scaling factor, and a word insertion penalty balances the tendency to output too many or too few words."""),
("Decoding and Search",
"""Decoding is the search for the word sequence that maximises the combined acoustic and language model score. The number of possible word sequences grows exponentially with utterance length, so exhaustive search is impossible.

Viterbi beam search keeps only the most promising partial hypotheses at each frame. Any hypothesis whose score falls more than a fixed margin, called the beam width, below the best hypothesis is pruned. A wider beam is more accurate but slower.

Weighted finite state transducers, or WFSTs, provide a clean way to combine the HMM topology, the context dependency, the lexicon and the grammar into a single search graph. The Kaldi toolkit popularised this approach. The composed graph is optimised with determinisation and minimisation so that search is efficient.

A decoder can output more than the single best path. An n-best list holds the top n hypotheses, and a lattice is a compact graph that stores many alternative hypotheses with their scores. Lattices are useful for rescoring with a stronger language model and for computing word confidence scores.

End-to-end models also need search. For attention based and transducer models, beam search over output tokens is used, with a typical beam size between 4 and 10. Whisper uses beam search with a beam size of 5 by default in many implementations, together with temperature fallback when the output looks unreliable."""),
("End-to-End Neural ASR: CTC, Attention and Transducers",
"""End-to-end models learn to map acoustic features directly to text with a single neural network trained on paired audio and transcripts, removing the need for a separate lexicon and HMM.

Connectionist temporal classification, or CTC, lets a network output one label per frame, including a special blank symbol. The CTC loss sums over every alignment that collapses to the correct transcript after merging repeated labels and removing blanks. CTC assumes outputs at different frames are conditionally independent, so it usually benefits from an external language model.

Attention based encoder-decoder models, such as Listen, Attend and Spell, use an encoder to process the whole utterance and a decoder that produces one token at a time while attending to the relevant encoder frames. They model dependencies between output tokens but need the full utterance before decoding, which makes streaming difficult.

The recurrent neural network transducer, or RNN-T, combines an encoder, a prediction network over previous tokens and a joint network. It supports streaming recognition and is widely used in on-device assistants.

Most end-to-end systems predict sub-word units produced by byte pair encoding or a unigram tokenizer. Sub-words keep the vocabulary small while still handling rare and new words.

Conformer encoders combine convolution, which captures local patterns, with self-attention, which captures long range context, and are a common choice in current systems."""),
("Transformers, Self-Supervised Learning and Whisper",
"""The transformer architecture relies on self-attention, which lets every position in a sequence attend to every other position. Because frames are processed in parallel, transformers train efficiently on large datasets.

Self-supervised learning reduces the need for transcribed audio. Wav2vec 2.0 masks parts of the latent speech representation and learns to identify the correct quantised representation among distractors. After pre-training on unlabelled audio, the model can be fine-tuned with a small amount of labelled data. HuBERT follows a similar idea using offline clustering targets.

Whisper, released by OpenAI in 2022, takes a different route called weak supervision. It is a transformer encoder-decoder trained on about 680,000 hours of audio paired with transcripts collected from the internet. The input is a 30 second window of 80 channel log mel spectrogram. Special tokens in the decoder select the task, such as transcription or translation, the language, and whether to predict timestamps.

Whisper accepts a text prompt that conditions the decoder. Supplying domain words in the prompt, which is called vocabulary biasing, can help the model spell technical terms correctly.

Whisper reports an average log probability and a no speech probability for each segment. Applications can use these values as confidence signals to decide whether to trust a transcript. Faster-whisper is a reimplementation using CTranslate2 that runs several times faster with lower memory use, including int8 quantisation on CPUs."""),
("Evaluating ASR: Word Error Rate",
"""The standard metric for speech recognition is the word error rate, or WER. The hypothesis is aligned with the reference transcript using minimum edit distance, and three kinds of errors are counted: substitutions, deletions and insertions.

WER equals the number of substitutions plus deletions plus insertions, divided by the number of words in the reference. Because insertions are counted, WER can be greater than 100 percent.

For example, if the reference is the cat sat on the mat and the hypothesis is the cat sat on mat, there is one deletion in six reference words, giving a WER of about 16.7 percent.

Text normalisation strongly affects WER. Before scoring, both reference and hypothesis should be lower-cased, punctuation removed and numbers written consistently. Otherwise a correct transcript can be penalised merely for writing 10 instead of ten.

Character error rate, or CER, applies the same formula to characters and is preferred for languages without clear word boundaries. Real time factor, the processing time divided by the audio duration, measures speed; a real time factor below one means faster than real time.

Results should be reported separately for different conditions, such as clean versus noisy audio or general versus technical vocabulary, because an average can hide large differences. Confidence intervals or at least the number of test utterances should always be stated."""),
("Challenges and Accessibility Applications",
"""Despite large improvements, ASR still struggles in several conditions. Background noise and reverberation reduce accuracy, especially with distant microphones. Overlapping speakers are difficult to separate. Accents and dialects that are under-represented in training data lead to higher error rates, which raises fairness concerns.

Domain specific vocabulary such as medical, legal or technical terms is another challenge. General models may mis-spell rare words, which is why vocabulary biasing and custom language models are used.

Speech technology has important accessibility uses. Text-to-speech, or TTS, reads text aloud for blind and low vision users. Modern neural TTS systems generate natural sounding speech from text using an acoustic model that predicts a mel spectrogram and a vocoder that converts it into a waveform. Speaking rate control lets listeners slow down or speed up the voice.

For dyslexic learners, listening to text while following along can reduce reading fatigue. Voice input lets students ask questions without typing. Combining ASR, retrieval of relevant passages and TTS makes it possible to build study assistants that answer questions about a student's own notes.

Such systems must be designed carefully. They should state where an answer comes from, admit when the notes do not contain an answer, and protect the privacy of uploaded documents. Clear spoken feedback, large high contrast interfaces and keyboard access are essential."""),
]


def build(out_path: str):
    doc = pymupdf.open()
    doc.set_metadata({"title": "Speech Recognition Fundamentals: Study Notes"})
    for i, (heading, body) in enumerate(PAGES, start=1):
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text((50, 60), heading, fontsize=16, fontname="helv")
        page.insert_textbox(pymupdf.Rect(50, 80, 545, 790), body, fontsize=11, fontname="helv", lineheight=1.3)
        page.insert_text((280, 820), f"Page {i}", fontsize=9, fontname="helv")
    doc.save(out_path)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "speech_recognition_notes.pdf")
    build(out)
    print("wrote", out)
