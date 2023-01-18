import * as tf from '@tensorflow/tfjs';

class MultiHeadAttention extends tf.layers.Layer {
	constructor(nState, nHead) {
		super({});
		// this.nState = nState;
		this.nHead = nHead;
		this.query = tf.layers.dense({ inputShape: nState, units: nState });
		this.key = tf.layers.dense({ inputShape: nState, units: nState, useBias: false });
		this.value = tf.layers.dense({ inputShape: nState, units: nState });
		this.out = tf.layers.dense({ inputShape: nState, units: nState });
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
	constructor(nState, nHead, crossAttention = false) {
		super();
		this.attn = new MultiHeadAttention(nState, nHead);
		this.attn_ln = tf.layers.layerNormalization({ inputShape: nState });

		this.cross_attn = crossAttention ? new MultiHeadAttention(nState, nHead) : null;
		this.cross_attn_ln = crossAttention ? tf.layers.layerNormalization({ inputShape: nState }) : null;

		const nMlp = nState * 4;
		this.mlp1 = tf.layers.dense({ inputShape: nState, units: nMlp });
		this.mlp2 = new GeLU();
		this.mlp3 = tf.layers.dense({ inputShape: nMlp, units: nState });
		this.mlpLn = tf.layers.layerNormalization({ inputShape: nState });
	}

	call({ x, xa, mask, kvCache }) {
		let xLayerNormed = this.attn_ln.apply(x);
		x = x.add(this.attn.apply({ x: xLayerNormed, mask: mask, kvCache: kvCache }));
		if (this.cross_attn) {
			x = x.add(this.cross_attn.apply({ x: this.cross_attn_ln.apply(x), xa: xa, kvCache: kvCache }));
		}
		x = this.mlpLn.apply(x);
		x = this.mlp1.apply(x);
		x = this.mlp2.apply(x);
		x = this.mlp3.apply(x);
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
	constructor(nMels, n_ctx, nState, nHead, n_layer) {
		super();
		this.gelu = new GeLU();
		this.conv1 = tf.layers.conv1d({ filters: nState, kernelSize: 3, padding: 'same' });
		this.conv2 = tf.layers.conv1d({ filters: nState, kernelSize: 3, strides: 2, padding: 'same' });

		this.positionalEmbedding = sinusoids(n_ctx, nState);
		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock(nState, nHead, true));
		}
		this.ln_post = tf.layers.layerNormalization({ inputShape: nState });
	}

	call(x) {
		let xTransposed = x.transpose([ 0, 2, 1 ]);
		x = this.gelu.apply(this.conv1.apply(xTransposed));
		x = this.gelu.apply(this.conv2.apply(x));
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
	constructor(nVocab, n_ctx, nState, nHead, n_layer) {
		super();
		this.tokenEmbedding = tf.layers.embedding({
			inputDim: nVocab,
			outputDim: nState,
			trainable: false
		});
		this.positionalEmbedding = tf.zeros([ n_ctx, nState ]);
		this.blocks = [];
		for (let i = 0; i < n_layer; i++) {
			this.blocks.push(new ResidualAttentionBlock(nState, nHead, true));
		}
		this.ln = tf.layers.layerNormalization({ inputShape: nState });

		const triuMask = tf.fill([ n_ctx, n_ctx ], -Infinity).arraySync().map((array, i) => {
			return array.map((num, j) => {
				if (j > i) return 0;
				return num;
			});
		});
		// TODO: disable gradients for mask and somehow add to "state dict"
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
		weights = tf.transpose(this.tokenEmbedding.getWeights()[0], [ 1, 0 ]);
		x = x.matMul(weights);
		return x;
	}

	static get className() {
		return 'TextDecoder';
	}
}

export class Whisper extends tf.layers.Layer {
	constructor(dims) {
		super();
		this.dims = dims;
		// assume dims is dict-like object
		// can we access with '.' ?
		this.encoder = new AudioEncoder(
			this.dims.nMels,
			this.dims.nAudioCtx,
			this.dims.nAudioState,
			this.dims.nAudioHead,
			this.dims.nAudioLayer
		);
		this.decoder = new TextDecoder(
			this.dims.nVocab,
			this.dims.nTextCtx,
			this.dims.nTextState,
			this.dims.nTextHead,
			this.dims.nTextLayer
		);
	}

	static get className() {
		return 'Whisper';
	}

	embed_audio(mel) {
		return this.encoder.apply(mel);
	}

	logits(tokens, audio_features) {
		this.decoder.apply(tokens, audio_features);
	}

	call(mel, tokens) {
		this.decoder.apply(tokens, this.encoder(mel));
	}
}
