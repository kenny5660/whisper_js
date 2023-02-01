const tf = require('@tensorflow/tfjs');

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

class GreedyDecoder {
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
		const currentLogprobs = tf.gather(logprobs, nextTokens, -1);
		const tokensLastShapeIndex = tokens.shape.length - 1;

		sumLogprobs = sumLogprobs.add(currentLogprobs.mul(tf.notEqual(tokens.slice(tokensLastShapeIndex), this.eot)));
		nextTokens = tf.where(
			tf.equal(tokens.slice(tokensLastShapeIndex), this.eot).transpose(),
			tf.zerosLike(nextTokens).add(this.eot),
			nextTokens
		);
		tokens = tokens.concat(nextTokens, -1);

		const completed = tf.equal(tokens.slice(tokensLastShapeIndex), this.eot).all();
		return [ tokens, completed ];
	}

	finalize(tokens, sumLogprobs) {
		// tokens = tokens.concat(tf.fill([ tokens.shape[0], 1 ], this.eot), 1);
		tokens = tf.pad3d(tokens, [ [ 0, 0 ], [ 0, 0 ], [ 0, 1 ] ], this.eot);
		return [ tokens, sumLogprobs.dataSync() ];
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
