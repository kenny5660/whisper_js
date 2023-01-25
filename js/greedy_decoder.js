const tf = require('@tensorflow/tfjs')

class GreedyDecoder {
    constructor(temperature, eot) {
        this.temperature = temperature;
        this.eot = eot;
        this.sumLogprobs = sumLogprobs;
    }

    update(tokens, logits) {
        let nextTokens;
        if (this.temperature === 0) {
            nextTokens = tf.argMax(logits, -1);
        } else {
            nextTokens = tf.multinomial(logits.div(this.temperature), 1);
        }

        const logprobs = tf.logSoftmax(logits);
        const currentLogprobs = tf.gather(logprobs, nextTokens, -1);
        const tokensLastShapeIndex = tokens.shape.length - 1
        this.sumLogprobs = this.sumLogprobs.add(
            currentLogprobs.mul(tf.notEqual(tokens.slice(tokensLastShapeIndex), this.eot))
        );
        nextTokens = tf.where(
            tf.equal(tokens.slice(tokensLastShapeIndex), this.eot).transpose(),
            tf.zerosLike(nextTokens).add(this.eot), nextTokens
        );
        tokens = tokens.concat(nextTokens, -1);

        const completed = tf.equal(tokens.slice(tokensLastShapeIndex), this.eot).all()
        return [tokens, completed];
    }

    finalize(tokens) {
        tokens = tokens.concat(tf.fill([tokens.shape[0], 1], this.eot), 1);
        // tokens = tf.pad(tokens, [[0, 1]], this.eot)
        return [tokens, this.sumLogprobs.dataSync()];
    }
}
