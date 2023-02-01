import * as tf from '@tensorflow/tfjs';


export class GreedyDecoder {
	constructor(temperature, eot) {
		this.temperature = temperature;
		this.eot = eot;
		// this.sumLogprobs = sumLogprobs;
	}

	update(tokens, logits, sumLogprobs) {
		let nextTokens;
		if (this.temperature === 0) {
			nextTokens = tf.argMax(logits, -1);
		} else {
			nextTokens = tf.multinomial(logits.div(this.temperature), 1);
		}

		const logprobs = tf.logSoftmax(logits);
		const currentLogprobs = tf.gather(logprobs, nextTokens, -1); // TODO temp === 0
		let tokensLastShapeIndex = tokens.shape[tokens.shape.length - 1] - 1;
		const tokensSlice = tokens.slice([0, tokensLastShapeIndex], [tokens.shape[0], 1]);
		sumLogprobs = sumLogprobs.add(currentLogprobs.mul(tf.notEqual(tokensSlice, this.eot)));
		nextTokens = tf.where(
			tf.equal(tokensSlice, this.eot).transpose(),
			tf.zerosLike(nextTokens).add(this.eot),
			nextTokens
		);
		tokens = tokens.concat(nextTokens, -1);
		
		tokensLastShapeIndex = tokens.shape[tokens.shape.length - 1] - 1;
		// let completed = tf.equal(tokens.slice([0, tokensLastShapeIndex], [tokens.shape[0], 1]), this.eot).all();
		let completed = tokens.slice([0, tokensLastShapeIndex], [tokens.shape[0], 1]).equal(this.eot).dataSync()[0];
		return [ tokens, completed ];
	}

	finalize(tokens, sumLogprobs) {
		//tokens rank == 3
		// tokens = tokens.concat(tf.fill([ tokens.shape[0], 1 ], this.eot), 1);
		tokens = tf.pad3d(tokens, [ [ 0, 0 ], [ 0, 0 ], [ 0, 1 ] ], this.eot);
		// tokens = tf.pad(tokens, [[0, 1]], this.eot)
		return [ tokens, sumLogprobs.dataSync() ];
	}
}
