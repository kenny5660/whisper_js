import * as tf from '@tensorflow/tfjs';


/*
Жадный декодер или выбрать наиболее вероятный токен.
constructor:
- temperature или temperature scaling - чем меньше, добавляет меньше уверенности, но больше разнообразия
https://arxiv.org/pdf/1706.04599.pdf
- eot токен end of sequence
update:
На входе: токены, вероятности токенов и накопленные вероятности( на каждый такт)
Текущее вероятности складываем с накопленными вероятностями, но только для тех у которых != eot.
Соединяем текущие токены с предсказанными токенами по -1.
Если последний токен совпадает с символом eot, тогда completed
finalize:
Заполняет оставшуюся последовательность eot
 */

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
		return [ tokens, sumLogprobs.arraySync() ];
	}
}

//
// const tokens = tf.tensor([[1, 2], [3, 4]]);
// const logits = tf.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]);
// const sumLogprobs = tf.scalar(0);
//
// const decoder = new GreedyDecoder(0.5, 3);
//
// let [tokens_updated, completed] = decoder.update(tokens, logits, sumLogprobs);
// console.log("Tokens:", tokens_updated.dataSync());
// console.log("Completed:", completed);
//
// [tokens_finalized, sumLogprobs_finalized] = decoder.finalize(tokens, sumLogprobs);
// console.log("Final Tokens:", tokens_finalized.dataSync());
// console.log("Final Logprobs:", sumLogprobs_finalized);
