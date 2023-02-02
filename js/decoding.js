import * as tf from '@tensorflow/tfjs';
import { GreedyDecoder } from './greedy_decoder.js';
import { MaximumLikelihoodRanker } from './maximum_likehood_ranker.js';
import { getTokenizer } from './tokenizer/tokenizer.js';

export class DecodingResult {
	constructor({ language, text = '' } = {}) {
		this.language = language;
		this.text = text;
	}
}

function assignColumns(tensor, cols, value) {
	let buffer = tensor.bufferSync();
	for (let row = 0; row < buffer.shape[0]; row++) {
		for (let col of cols) {
			buffer.set(value, row, col);
		}
	}
	return buffer.toTensor();
}
class SuppressTokens {
	constructor(suppressTokens) {
		this.suppressTokens = [ ...suppressTokens ];
	}

	apply(logits, tokens) {
		// logits.rank == 2
		const res = assignColumns(logits, this.suppressTokens, -1e18);
		return [ res, tokens ];
	}
}

export class DecodingTask {
	constructor(model, options) {
		this.model = model;
		let language = options.language ? options.language : 'en';
		console.log(getTokenizer);
		let tokenizer = getTokenizer(model.isMultilingual(), options.task, language);
		this.tokenizer = tokenizer;
		this.options = this.verifyOptions(options);

		this.nGroup = options.beamSize ? options.beamSize : 1; // temp === 0
		this.nCtx = model.dims.get('n_text_ctx').value;
		this.sampleLen = options.sampleLen ? options.sampleLen : Math.floor(this.nCtx / 2);

		this.sotSequence = [ ...tokenizer.sotSequence ];
		if (this.options.withoutTimestamps) {
			this.sotSequence = [ ...tokenizer.sotSequence, tokenizer.noTimestamps ];
		}
		this.initialTokens = this.getInitialTokens();
		this.sampleBegin = this.initialTokens.length;
		this.sotIndex = this.initialTokens.indexOf(this.tokenizer.sot);
		this.sequenceRanker = new MaximumLikelihoodRanker(options.lengthPenalty);

		if (options.beamSize) {
			// this.decoder = new BeamSearchDecoder();
		} else {
			this.decoder = new GreedyDecoder(options.temperature, tokenizer.eot);
		}
		this.logitFilters = [];

		if (this.options.suppressTokens) {
			this.logitFilters.push(new SuppressTokens(this.getSuppressTokens()));
		}
	}

	verifyOptions(options) {
		if (options.beamSize !== null && options.bestOf !== null) {
			throw new Error("beam_size and best_of can't be given together");
		}
		if (options.patience !== null && options.beamSize === null) {
			throw new Error('patience requires beam_size to be given');
		}

		if (options.lengthPenalty !== null && !(0 <= options.lengthPenalty && options.lengthPenalty <= 1)) {
			throw new Error('length_penalty (alpha) should be a value between 0 and 1');
		}

		return options;
	}

	getInitialTokens() {
		let tokens = [ ...this.sotSequence ];
		return tokens;
	}

	getSuppressTokens() {
		let suppressTokens = Array.isArray(this.options.suppressTokens)
			? [ ...this.options.suppressTokens ]
			: this.options.suppressTokens;
		if (typeof suppressTokens === 'string') {
			suppressTokens = suppressTokens.split(',').map((x) => Number(x));
		}
		// if -1 => in array
		if (suppressTokens.indexOf(-1) !== -1) {
			suppressTokens = suppressTokens.filter((x) => x >= 0);
			suppressTokens = [ ...suppressTokens, ...this.tokenizer.nonSpeechTokens ];
		} else {
			if (!suppressTokens || suppressTokens.length === 0) suppressTokens = [];
			else throw new Error('suppress_tokens must be a list');
		}
		suppressTokens.push(this.tokenizer.sot);
		suppressTokens.push(this.tokenizer.sotPrev);
		suppressTokens.push(this.tokenizer.sotLm);
		if (!this.tokenizer.noSpeech) {
			suppressTokens.push(this.tokenizer.noSpeech);
		}
		return [ ...new Set(suppressTokens) ].sort((a, b) => a - b);
	}

	getAudioFeatures(mel) {
		let audioFeatures;
		if (
			mel.shape[mel.shape.length - 2] === this.model.dims.get('n_audio_ctx').value &&
			mel.shape[mel.shape.length - 1] === this.model.dims.get('n_audio_state').value
		) {
			audioFeatures = mel;
		} else {
			audioFeatures = this.model.encoder.apply(mel);
		}
		return audioFeatures;
	}

	detectLanguage(audioFeatures, tokens) {
		let languages = Array(audioFeatures.shape[0]).fill(this.options.language);
		let langTokens = [];
		let langProbs = [];
		if (!this.options.language) {
			throw new Error('specify language');
		}
		return languages, langProbs;
	}

	mainLoop(audioFeatures, tokens) {
		if (audioFeatures.shape[0] !== tokens.shape[0]) throw new Error('Number of feature tensors differs');
		const nBatch = tokens.shape[0];
		let sumLogprobs = tf.zeros([ nBatch ]);
		let noSpeechProb = Array(nBatch).fill(null);
		for (let i = 0; i < this.sampleLen; i++) {
			let logits = this.model.logits(tokens, audioFeatures);
			if (i === 0 && this.tokenizer.noSpeech) {
				let probsAtSot = [];
				for (let j = 0; j < logits.shape[0]; j++) {
					probsAtSot.push(logits.gather(j).gather(this.sotIndex).arraySync());
				}
				probsAtSot = tf.tensor(probsAtSot).cast('float32');
				probsAtSot = tf.softmax(probsAtSot);
				let probsAtSotT = tf.transpose(probsAtSot);
				noSpeechProb = tf.transpose(probsAtSotT.gather(this.tokenizer.noSpeech)).arraySync();
			}
			logits = logits.gather(logits.shape[1] - 1, 1);
			for (let logitFilter of this.logitFilters) {
				[ logits, tokens ] = logitFilter.apply(logits, tokens);
			}
			let completed;
			[ tokens, completed ] = this.decoder.update(tokens, logits, sumLogprobs);
			if (completed || tokens.shape[tokens.shape.length - 1] > this.nCtx) break;
			console.log(i);
			console.log(tokens.shape);
		}
		return [ tokens, sumLogprobs, noSpeechProb ];
	}

	run(mel) {
		let tokenizer = this.tokenizer;
		let nAudio = mel.shape[0];

		let audioFeatures = this.getAudioFeatures(mel);
		let tokens = tf.tensor2d(Array(nAudio).fill(this.initialTokens));

		let [ languages, languageProbs ] = this.detectLanguage(audioFeatures, tokens);
		// TODO check shapes
		audioFeatures = tf.tensor(Array(this.nGroup).fill(audioFeatures.arraySync()[0]));
		tokens = tf.tensor(Array(this.nGroup).fill(tokens.arraySync()[0]));
		let sumLogprobs, noSpeechProbs;
		[ tokens, sumLogprobs, noSpeechProbs ] = this.mainLoop(audioFeatures, tokens);
		console.log(sumLogprobs);

		audioFeatures = audioFeatures.gather(tf.range(0, audioFeatures.shape[0], this.nGroup, 'int32'));
		tokens = tokens.reshape([ nAudio, this.nGroup, -1 ]);
		sumLogprobs = sumLogprobs.reshape([ nAudio, this.nGroup ]);

		[ tokens, sumLogprobs ] = this.decoder.finalize(tokens, sumLogprobs);
		let lists = new Array();
		for (let i = 0; i < tokens.shape[0]; i++) {
			let s = tokens.gather(i);
			let list = new Array();
			for (let j = 0; j < s.shape[0]; j++) {
				let t = s.gather(j);
				let mask = t.equal([ tokenizer.eot ]).asType('bool');
				const endIdx = mask.arraySync().indexOf(1);
				let item = t.slice(this.sampleBegin, endIdx - this.sampleBegin).arraySync();
				list.push(item);
			}
			lists.push(list);
		}
		tokens = lists;
		let selected = this.sequenceRanker.rank(tokens, sumLogprobs);

		let newTokens = new Array();
		for (let i = 0; i < tokens.length; i++) {
			let idx = Math.floor(selected[i]);
			let t = tokens[i];
			newTokens.push(t[idx]);
		}

		let texts = new Array();
		const nonAlpha = '!.,?';
		for (let i = 0; i < newTokens.length; i++) {
			let t = newTokens[i];
			let words = tokenizer.decode(t);
			let string = '';
			words.forEach((word) => {
				if (nonAlpha.includes(word[0])) {
					string += word.trim();
				} else {
					string = string + ' ' + word.trim();
				}
			});
			texts.push(string.trim());
		}

		let results = new Array();
		for (let i = 0; i < audioFeatures.shape[0]; i++) {
			const text = texts[i];
			//const lang = languages[i];
			const lang = 'en';
			results.push(
				new DecodingResult({
					language: lang,
					text: text
				})
			);
		}
		return results;
	}
}
// module.exports = { DecodingTask };
