import * as tf from '@tensorflow/tfjs';
import { DecodingTask } from '../decoding.js';

class MultiHeadAttention extends tf.layers.Layer {
	constructor(nState, nHead, weights) {
		super({});
		// this.nState = nState;
		this.nHead = nHead;

		const prefix = Object.keys(weights)[0].split('.')[0]; // {'cross_attn', 'attn'}

		this.query = tf.layers.dense({
			inputShape: nState,
			units: nState,
			weights: [ weights[prefix + '.query.weight'], weights[prefix + '.query.bias'] ]
		});
		this.key = tf.layers.dense({
			inputShape: nState,
			units: nState,
			useBias: false,
			weights: [ weights[prefix + '.key.weight'] ]
		});
		this.value = tf.layers.dense({
			inputShape: nState,
			units: nState,
			weights: [ weights[prefix + '.value.weight'], weights[prefix + '.value.bias'] ]
		});
		this.out = tf.layers.dense({
			inputShape: nState,
			units: nState,
			weights: [ weights[prefix + '.out.weight'], weights[prefix + '.out.bias'] ]
		});
	}

	static get className() {
		return 'MultiHeadAttention';
	}

	call({ x, xa, mask, kvCache }) {
		const q = this.query.apply(x);
		let k, v;
		if (!kvCache || !xa || !kvCache[this.key]) {
			k = this.key.apply(!xa ? x : xa);
			v = this.value.apply(!xa ? x : xa);
		} else {
			k = kvCache[this.key];
			v = kvCache[this.value];
		}
		const wv = this.qkv_attention({ q: q, k: k, v: v, mask: mask });
		return this.out.apply(wv);
	}

	qkv_attention({ q, k, v, mask }) {
		let [ nBatch, nCtx, nState ] = q.shape;
		let scale = Math.pow(Math.floor(nState / this.nHead), -0.25);
		q = q.reshape([ ...q.shape.slice(0, 2), this.nHead, -1 ]).transpose([ 0, 2, 1, 3 ]).mul(scale);
		k = k.reshape([ ...k.shape.slice(0, 2), this.nHead, -1 ]).transpose([ 0, 2, 3, 1 ]).mul(scale);
		v = v.reshape([ ...v.shape.slice(0, 2), this.nHead, -1 ]).transpose([ 0, 2, 1, 3 ]);

		let qk = q.matMul(k);
		if (mask) {
			qk = qk.add(mask.slice([ 0, 0 ], [ nCtx, nCtx ]));
		}
		const axis = qk.shape.length - 1;
		qk = qk.cast('float32');
		const w = tf.layers.softmax({ axis: axis }).apply(qk);
		let result = w.matMul(v).transpose([ 0, 2, 1, 3 ]);

		const oldShape = result.shape;
		const flattenedShape = tf.prod(oldShape.slice(2));
		const newShape = tf.concat([ oldShape.slice(0, 2), tf.tensor1d([ flattenedShape.arraySync() ]) ]);

		result = result.reshape(newShape.arraySync());
		return result;
	}
}

class GeLU extends tf.layers.Layer {
	static get className() {
		return 'GeLU';
	}

	call(x) {
		let arg = x.div(Math.sqrt(2));
		let y = x.mul(tf.erf(arg).add(tf.scalar(1)).mul(0.5));
		return y;
	}
}

class ResidualAttentionBlock extends tf.layers.Layer {
	constructor(nState, nHead, weights, crossAttention = false) {
		super();
		this.attn = new MultiHeadAttention(nState, nHead, weights);
		this.attn_ln = tf.layers.layerNormalization({
			inputShape: nState,
			weights: [ weights['attn_ln.weight'], weights['attn_ln.bias'] ],
			trainable: false
		});

		this.cross_attn = crossAttention ? new MultiHeadAttention(nState, nHead, weights) : null;
		this.cross_attn_ln = crossAttention
			? tf.layers.layerNormalization({
					inputShape: nState,
					weights: [ weights['cross_attn_ln.weight'], weights['cross_attn_ln.bias'] ],
					trainable: false
				})
			: null;

		const nMlp = nState * 4;
		this.mlp1 = tf.layers.dense({
			inputShape: nState,
			units: nMlp,
			weights: [ weights['mlp.0.weight'], weights['mlp.0.bias'] ]
		});
		this.mlp2 = new GeLU();
		this.mlp3 = tf.layers.dense({
			inputShape: nMlp,
			units: nState,
			weights: [ weights['mlp.2.weight'], weights['mlp.2.bias'] ]
		});
		this.mlpLn = tf.layers.layerNormalization({
			inputShape: nState,
			weights: [ weights['mlp_ln.weight'], weights['mlp_ln.bias'] ],
			trainable: false
		});
	}

	call({ x, xa, mask, kvCache }) {
		let xLayerNormed = this.attn_ln.apply(x);
		x = x.add(this.attn.apply({ x: xLayerNormed, mask: mask, kvCache: kvCache }));
		if (this.cross_attn) {
			let arg = this.cross_attn.apply({ x: this.cross_attn_ln.apply(x), xa: xa, kvCache: kvCache });
			x = x.add(arg);
		}
		let xToAdd = this.mlpLn.apply(x);
		xToAdd = this.mlp1.apply(xToAdd);
		xToAdd = this.mlp2.apply(xToAdd);
		xToAdd = this.mlp3.apply(xToAdd);
		x = x.add(xToAdd);
		return x;
	}

	static get className() {
		return 'ResidualBlockAttention';
	}
}

function sinusoids(length, channels, max_timescale = 10000) {
	tf.util.assert(channels % 2 == 0);
	const log_timescale_increment = Math.log(max_timescale) / (Math.floor(channels / 2) - 1);
	const inv_timescales = tf.exp(tf.range(0, Math.floor(channels / 2)).mul(-log_timescale_increment));
	const m1 = tf.range(0, length).as2D(length, 1);
	const m2 = inv_timescales.as2D(1, inv_timescales.shape[0]);
	const scaled_time = m1.mul(m2);
	return tf.concat([ scaled_time.sin(), scaled_time.cos() ], 1);
}

class AudioEncoder extends tf.layers.Layer {
	constructor(nMels, n_ctx, nState, nHead, n_layer, weights) {
		super();
		this.gelu = new GeLU();

		this.conv1 = tf.layers.conv1d({
			filters: nState,
			kernelSize: 3,
			// padding: 'same',
			padding: 'valid',
			weights: [ weights['encoder.conv1.weight'], weights['encoder.conv1.bias'] ],
			dataFormat: 'channelsFirst'
		});

		this.conv2 = tf.layers.conv1d({
			filters: nState,
			kernelSize: 3,
			strides: 2,
			padding: 'valid',
			weights: [ weights['encoder.conv2.weight'], weights['encoder.conv2.bias'] ],
			dataFormat: 'channelsFirst'
		});
		this.conv1_weights = [ weights['encoder.conv1.weight'], weights['encoder.conv1.bias'] ];
		this.conv2_weights = [ weights['encoder.conv2.weight'], weights['encoder.conv2.bias'] ];
		this.positionalEmbedding = sinusoids(n_ctx, nState);
		// this.positionalEmbedding = weights['encoder.positional_embedding'];
		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock(nState, nHead, weights['encoder.blocks.'][i]));
		}
		this.ln_post = tf.layers.layerNormalization({
			inputShape: nState,
			weights: [ weights['encoder.ln_post.weight'], weights['encoder.ln_post.bias'] ],
			trainable: false
		});
	}

	call(x) {
		// x.shape == [1, 80, 3000]
		function pad(x) {
			let padding = tf.zeros([ 1, x.shape[1], 1 ]);
			return padding.concat(x.concat(padding, 2), 2);
		}
		x = pad(x);
		x = this.conv1.apply(x);
		x = this.gelu.apply(x);
		x = x.transpose([ 0, 2, 1 ]);
		x = pad(x);
		x = this.conv2.apply(x);
		x = this.gelu.apply(x);

		tf.util.assert(
			tf.equal(tf.tensor(x.shape.slice(1)), tf.tensor(this.positionalEmbedding.shape)),
			'incorrect audio shape'
		);
		x = x.add(this.positionalEmbedding).cast(x.dtype);
		for (let block of this.blocks) {
			x = block.apply({ x: x });
		}
		return this.ln_post.apply(x);
	}

	static get className() {
		return 'AudioEncoder';
	}
}

class TextDecoder extends tf.layers.Layer {
	constructor(nVocab, n_ctx, nState, nHead, n_layer, weights) {
		super();
		this.tokenEmbedding = tf.layers.embedding({
			inputDim: nVocab,
			outputDim: nState,
			trainable: false,
			weights: [weights['decoder.token_embedding.weight']]
		});
		// this.positionalEmbedding = tf.zeros([ n_ctx, nState ]);
		this.positionalEmbedding = weights['decoder.positional_embedding'];
		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock(nState, nHead, weights['decoder.blocks.'][i], true));
		}
		this.ln = tf.layers.layerNormalization({
			inputShape: nState,
			weights: [ weights['decoder.ln.weight'], weights['decoder.ln.bias'] ],
			trainable: false
		});

		const triuMask = tf.fill([ n_ctx, n_ctx ], -Infinity).arraySync().map((array, i) => {
			return array.map((num, j) => {
				if (j > i) return 0;
				return num;
			});
		});
		this.mask = tf.tensor(triuMask);
	}

	call(x, xa, kvCache) {
		let offset = 0;
		if (kvCache) {
			const values = Object.values(kvCache);
			offset = values[0].shape[1];
		}
		const xEmbeddedTokens = this.tokenEmbedding.apply(x);
		const firstDimLen = x.shape[x.shape.length - 1];
		const secondDimLen = this.positionalEmbedding.shape[this.positionalEmbedding.shape.length - 1];
		const positionalEmbedding = this.positionalEmbedding.slice(
			[ offset, 0 ],
			[ offset + firstDimLen, secondDimLen ]
		);
		x = xEmbeddedTokens.add(positionalEmbedding);
		x = x.cast(xa.dtype);
		for (let block of this.blocks) {
			x = block.apply({ x: x, xa: xa, mask: this.mask, kvCache: kvCache });
		}
		x = this.ln.apply(x);

		let weights = this.tokenEmbedding.getWeights()[0];
		weights = tf.transpose(weights, [ 1, 0 ]);
		x = x.matMul(weights);
		return x;
	}

	static get className() {
		return 'TextDecoder';
	}
}

export class Whisper extends tf.layers.Layer {
	constructor(weights) {
		super();
		this.dims = weights.get('dims');
		this.model_state_dict = weights.get('model_state_dict');

		let encoderWeights = { 'encoder.blocks.': {} };
		let decoderWeights = { 'decoder.blocks.': {} };
		let self = this;
		function collectBlockWeights(fullName, prefix, weights) {
			const num = Number(fullName[prefix.length]);

			// TODO: only work for numbers < 10
			const attn_layer_name = fullName.substring(prefix.length + 2);

			if (typeof weights[prefix][num] === 'undefined') {
				weights[prefix][num] = {};
			}
			let dataset = self.model_state_dict.get(fullName);
			weights[prefix][num][attn_layer_name] = tf.tensor(dataset.value, dataset.shape);
		}

		for (let name of this.model_state_dict.keys()) {
			if (name.includes('encoder')) {
				if (!name.includes('blocks')) {
					let dataset = this.model_state_dict.get(name);
					encoderWeights[name] = tf.tensor(dataset.value, dataset.shape);
				} else {
					collectBlockWeights(name, 'encoder.blocks.', encoderWeights);
				}
			}
			if (name.includes('decoder')) {
				if (!name.includes('blocks')) {
					let dataset = this.model_state_dict.get(name);
					decoderWeights[name] = tf.tensor(dataset.value, dataset.shape);
				} else {
					collectBlockWeights(name, 'decoder.blocks.', decoderWeights);
				}
			}
		}

		this.encoder = new AudioEncoder(
			this.dims.get('n_mels').value,
			this.dims.get('n_audio_ctx').value,
			this.dims.get('n_audio_state').value,
			this.dims.get('n_audio_head').value,
			this.dims.get('n_audio_layer').value,
			encoderWeights
		);
		this.decoder = new TextDecoder(
			this.dims.get('n_vocab').value,
			this.dims.get('n_text_ctx').value,
			this.dims.get('n_text_state').value,
			this.dims.get('n_text_head').value,
			this.dims.get('n_text_layer').value,
			decoderWeights
		);
	}

	static get className() {
		return 'Whisper';
	}

	embed_audio(mel) {
		const audioFeatures = this.encoder.apply(mel);
		console.log(audioFeatures.dataSync());
		return audioFeatures;
	}

	logits(tokens, audio_features) {
		return this.decoder.apply(tokens, audio_features);
	}

	call(mel, tokens) {
		this.decoder.apply(tokens, this.encoder(mel));
	}

	
	decode(mel, options) {
		options = {
			'task': 'transcribe',
			'language': 'en',
			'temperature': 0.0,
			'sampleLen': null,
			'bestOf': null,
			'beamSize': null,
			'patience': null,
			'lengthPenalty': null,
			'prompt': null,
			'prefix': null,
			'suppressBlank': true,
			'suppressTokens': '-1',
			'withoutTimestamps': true,
			'maxInitialTimestamp': 1.0,
		}
		const single = mel.shape.length === 2;
		if (single) mel = mel.expandDims(0);
		let result = new DecodingTask(this, options).run(mel);
		if (single) result = result[0];
		return result;
	}

	isMultilingual() {
		return this.dims.get('n_vocab').value == 51865;
	}

}
