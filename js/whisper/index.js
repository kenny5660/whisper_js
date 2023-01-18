import * as tf from '@tensorflow/tfjs';
import { Whisper } from './model.js';

// config of 'tiny' version
let model = new Whisper({
	nMels: 80,
	nAudioCtx: 1500,
	nAudioState: 384,
	nAudioHead: 6,
	nAudioLayer: 4,
	nVocab: 51865,
	nTextCtx: 448,
	nTextState: 384,
	nTextHead: 6,
	nTextLayer: 4
});

console.log(tf);
let mel_tensor = tf.ones([ 1, 80, 3000 ]);
let audio_features = model.embed_audio(mel_tensor);
console.log(audio_features.shape);
// tf.assert(audio_features.shape == [ 1, 1500, 384 ]);

let tokens = tf.tensor2d([ [ 50258 ] ]);
let decoding_result = model.decoder.apply(tokens, audio_features);
console.log(decoding_result.shape);
// tf.assert(decoding_result.shape == [ 1, 1, 51865 ]);
