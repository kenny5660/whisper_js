import * as tf from '@tensorflow/tfjs';
import { maxProb, assignColumn } from './utils.js';
import { GreedyDecoder, MaximumLikelihoodRanker } from './greedy_decoder.js';

// const tf = require('@tensorflow/tfjs');
// const { assignColumn } = require('./utils.js');
// const getTokenizer = require('./tokenizer.js');
// const { GreedyDecoder, MaximumLikelihoodRanker } = require('./decoders.js');

class DecodingResult {
	constructor(
		{
			audioFeatures,
			language,
			languageProbs = {},
			tokens = new Array(),
			text = '',
			avgLogprob = null,
			noSpeechProb = null,
			temperature = null,
			compressionRatio = null
		} = {}
	) {
		this.audioFeatures = audioFeatures;
		this.language = language;
		this.languageProbs = languageProbs;
		this.tokens = tokens;
		this.text = text;
		this.avgLogprob = avgLogprob;
		this.noSpeechProb = noSpeechProb;
		this.temperature = temperature;
		this.compressionRatio = compressionRatio;
	}
}

class SuppressTokens {
	constructor(suppressTokens) {
		this.suppressTokens = [ ...suppressTokens ];
	}

	apply(logits, tokens) {
		return [ assignColumn(logits, this.suppressTokens, Infinity), tokens ];
	}
}

class DecodingTask {
	constructor(model, options) {
		this.model = model;
		let language = options.language ? options.language : 'en';
		let tokenizer = getTokenizer(model.isMultilingual(), options.task, language);
		this.tokenizer = tokenizer;
		this.options = this.verifyOptions(options);

		this.nGroup = options.beamSize ? options.beamSize : options.bestOf ? options.bestOf : 1;
		this.nCtx = model.dims.nTextCtx;
		this.sampleLen = options.sampleLen ? options.sampleLen : Math.floor(model.dims.nTextCtx / 2);

		this.sotSequence = [ ...tokenizer.sotSequence ];
		if (this.options.withoutTimestamps) {
			// this.sotSequence = tokenizer.sotSequenceIncludingNotimestamps;
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
		if (this.options.suppressBlank) {
			//default is false
		}
		if (this.options.suppressTokens) {
			this.logitFilters.push(new SuppressTokens(this.getSuppressTokens()));
		}
		if (!options.withoutTimestamps) {
			//we set options.withoutTimestamps to false
			// const precision = CHUNK_LENGTH / model.dims.nAudioCtx;
			// let maxInitialTimestampIndex;
			// if (options.maxInitialTimestamp) {
			// 	maxInitialTimestampIndex = Math.round(this.options.maxInitialTimestamp / precision);
			// }
			// this.logitFilters.push(new ApplyTimestampRules(tokenizer, this.sampleBegin, maxInitialTimestampIndex));
		}
	}

	verifyOptions(options) {
		// if (options.beamSize !== null && options.bestOf !== null) {
		// 	throw new Error("beam_size and best_of can't be given together");
		// }
		// if (options.temperature === 0) {
		// 	if (options.bestOf !== null) {
		// 		throw new Error('best_of with greedy sampling (T=0) is not compatible');
		// 	}
		// }

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
		let prefix = this.options.prefix;
		let prompt = this.options.prompt;
		if (prefix) {
			// default prefix == None
		}
		if (prompt) {
			// default prompt == None
		}
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
			mel.shape[mel.shape.length - 2] === this.model.dims.nAudioCtx &&
			mel.shape[mel.shape.length - 1] === this.model.dims.nAudioState
		) {
			audioFeatures = mel;
		} else {
			// throw new Error('input is mel only now');
			// TODO
			audioFeatures = this.model.encoder.apply(mel);
		}
		return audioFeatures;
	}

	detectLanguage(audioFeatures, tokens) {
		let languages = Array(audioFeatures.shape[0]).fill(this.options.language);
		let langTokens = [];
		let langProbs = [];
		if (!this.options.language || this.options.task === 'lang_id') {
			throw new Error('specify language');
			// [ langTokens, langProbs ] = this.model.detectLanguage(audioFeatures, this.tokenizer);
			// maxProb(langProbs);
			// for (let i = 0; i < languages.length; i++) {
			// 	languages[i] = maxProb(langProbs)[0];
			// }
			// NO IDEA
			// if (!this.options.language) tokens =
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
				// let probsAtSot = logits([:, this.sotIndex ]).cast('float32');
				probsAtSot = tf.layers.softmax({ axis: -1 }).apply(probsAtSot);
				// probsAtSot is 2d now
				// noSpeechProb = probsAtSot.slice([ 0, 0 ], [ probsAtSot.shape[0], this.tokenizer.noSpeech ]).arraySync();
				let probsAtSotT = tf.transpose(probsAtSot);
				// no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()
				// let data = [];
				// for (let j of this.tokenizer.noSpeech.arraySync()) {
				// 	data.push(probsAtSotT.gather(j).arraySync());
				// }
				noSpeechProb = tf.transpose(probsAtSotT.gather(this.tokenizer.noSpeech)).arraySync();

				logits = logits.gather(logits.shape[1] - 1, 1);
				for (let logitFilter of this.logitFilters) {
					[ logits, tokens ] = logitFilter.apply(logits, tokens);
				}
				let completed;
				[ tokens, completed ] = this.decoder.update(tokens, logits, sumLogprobs);
				if (completed || tokens.shape[tokens.shape.length - 1] > n_cts) break;
			}
		}
		return [ tokens, sumLogprobs, noSpeechProb ];
	}

	run(mel) {
		let tokenizer = this.tokenizer;
		let nAudio = mel.shape[0];

		let audioFeatures = this.getAudioFeatures(mel);
		let tokens = tf.tensor2d(Array(nAudio).fill(this.initialTokens));

		let [ languages, languageProbs ] = this.detectLanguage(audioFeatures, tokens);
		if (this.options.task === 'lang_id') {
			//TODO
		}
		// TODO check shapes
		audioFeatures = tf.tensor(Array(this.nGroup).fill(audioFeatures.arraySync()[0]));
		tokens = tf.tensor(Array(this.nGroup).fill(tokens.arraySync()[0]));
		let sumLogprobs, noSpeechProbs;
		[ tokens, sumLogprobs, noSpeechProbs ] = this.mainLoop(audioFeatures, tokens);

		audioFeatures = audioFeatures.gather(tf.range(0, audioFeatures.shape[0], this.nGroup, 'int32'));
		noSpeechProbs = tf.tensor(noSpeechProbs);
		noSpeechProbs = noSpeechProbs.gather(tf.range(0, noSpeechProbs.shape[0], this.nGroup, 'int32'));
		noSpeechProbs = noSpeechProbs.arraySync();

		tokens = tokens.reshape([ nAudio, this.nGroup, -1 ]);
		sumLogprobs = sumLogprobs.reshape([ nAudio, this.nGroup ]);

		[ tokens, sumLogprobs ] = this.decoder.finalize(tokens, sumLogprobs);
		// let lists = new Array();
		// for (let i = 0; i < tokens.shape[0]; i++) {
		// 	let s = tokens.gather(i);
		// 	let list = new Array();
		// 	for (let j = 0; j < s.shape[0]; j++) {
		// 		let t = s.gather(j);
		// 		let mask = t.equal([ tokenizer.eot ]).asType('bool');
		// 		const endIdx = tf.where(mask).flatten().gather(0);
		// 		let item = t.slice(this.sampleBegin, endIdx);
		// 		list.push(item);
		// 	}
		// 	lists.push(list);
		// }

		let selected = this.sequenceRanker.rank(tokens, sumLogprobs);

		let newTokens = new Array();
		for (let i = 0; i < tokens.shape[0]; i++) {
			let idx = selected[i]; //to int
			let t = tokens.gather(i);
			newTokens.push(t.gather(idx));
		}

		let texts = new Array();
		for (let i = 0; i < newTokens.shape[0]; i++) {
			let t = newTokens[i];
			let str = tokenizer.decode(t);
			texts.push(str.trim());
		}

		// let newSumLogprobs = new Array();
		// for (let i = 0; i < sumLogprobs.length; i++) {
		// 	let idx = selected[i]; //to int
		// 	let lp = sumLogprobs[i];
		// 	newSumLogprobs.push(lp[i]);
		// }

		// let newAvgLogprobs = new Array();
		// for (let i = 0; i < sumLogprobs.length; i++) {
		// 	let t = tokens.gather(i);
		// 	let lp = sumLogprobs[i];
		// 	newSumLogprobs.push(lp / (t.shape[0] + 1));
		// }

		let results = new Array();
		for (let i = 0; i < audio_features.shape[0]; i++) {
			const text = texts[i];
			const lang = languages[i];
			// const tokens = newTokens[i];
			// const af = audioFeatures.gather(i);
			// const lp = newAvgLogprobs[i];
			// const nsp = noSpeechProbs[i];
			const tokens = null;
			const af = null;
			const lp = null;
			const nsp = null;
			results.push(
				new DecodingResult({
					audioFeatures: af,
					language: lang,
					tokens: tokens,
					text: text,
					avgLogprob: lp,
					noSpeechProb: nsp,
					temperature: this.options.temperature
				})
			);
		}

		return results;
	}
}
module.exports = { DecodingTask };
