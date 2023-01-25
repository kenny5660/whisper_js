const tf = require('@tensorflow/tfjs')

class DecodingOptions {
    constructor(options) {
        this.task = options.task || "transcribe";
        this.language = options.language;
        this.temperature = options.temperature || 0.0;
        this.sample_len = options.sample_len;
        this.best_of = options.best_of;
        this.beam_size = options.beam_size;
        this.patience = options.patience;
        this.length_penalty = options.length_penalty;
        this.prompt = options.prompt;
        this.prefix = options.prefix;
        this.suppress_blank = options.suppress_blank || true;
        this.suppress_tokens = options.suppress_tokens || "-1";
        this.without_timestamps = options.without_timestamps || false;
        this.max_initial_timestamp = options.max_initial_timestamp;
        this.fp16 = options.fp16 || true;
    }
}

class DecodingResult {
    constructor(result) {
        this.audio_features = result.audio_features;
        this.language = result.language;
        this.language_probs = result.language_probs;
        this.tokens = result.tokens || [];
        this.text = result.text || "";
        this.avg_logprob = result.avg_logprob || NaN;
        this.no_speech_prob = result.no_speech_prob || NaN;
        this.temperature = result.temperature || NaN;
        this.compression_ratio = result.compression_ratio || NaN;
    }
}


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
        this.sumLogprobs = this.sumLogprobs.add(currentLogprobs.mul(tf.notEqual(tokens.slice(tokensLastShapeIndex), this.eot)));
        nextTokens = tf.where(tf.equal(tokens.slice(tokensLastShapeIndex), this.eot).transpose(), tf.zerosLike(nextTokens).add(this.eot), nextTokens);
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
