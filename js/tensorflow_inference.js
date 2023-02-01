class TensorFlowJSInference {
    constructor(model, initialTokenLength) {
        this.model = model;
        this.initialTokenLength = initialTokenLength;
        this.kvCache = {};
        this.hooks = [];
    }

    logits(tokens, audioFeatures) {
        if (Object.keys(this.kvCache).length === 0) {
            [this.kvCache, this.hooks] = this.model.installKvCacheHooks();
        }

        if (tokens.shape[tokens.shape[0] - 1] > this.initialTokenLength) {
            tokens = tokens.slice(tokens.shape[0] - 1, tokens.shape[0]);
        }

        return this.model.decoder(tokens, audioFeatures, this.kvCache);
    }

    cleanupCaching() {
        this.kvCache = {};
        this.hooks = [];
    }

    rearrangeKVCache(sourceIndices) {
        for (let key in this.kvCache) {
            this.kvCache[key] = this.kvCache[key].gather(sourceIndices);
        }
    }
}


/*
const model = ...;
const initialTokenLength = 10;
const inference = new TensorFlowJSInference(model, initialTokenLength);

const tokens = tf.tensor2d([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]);
const audioFeatures = tf.tensor2d([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]);

const logits = inference.logits(tokens, audioFeatures);
console.log(logits.dataSync());

inference.cleanupCaching();
 */